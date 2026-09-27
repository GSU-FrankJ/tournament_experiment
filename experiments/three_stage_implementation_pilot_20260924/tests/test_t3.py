"""Unit tests for the T=3 implementation pilot (run with python -B -m unittest)."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import torch

E_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E_DIR))

import common  # noqa: E402,F401
import build_manifests as bm  # noqa: E402
import collect_t3  # noqa: E402
import make_report  # noqa: E402
import metrics  # noqa: E402
import run_experiment as rx  # noqa: E402

import utils.dp_br_verifier as dpv  # noqa: E402
from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.run_final_dp_br import collect_batch as w_collect_batch  # noqa: E402
from run.run_final_dp_br import make_policy_fns, run_verifier  # noqa: E402
from utils.theory_multistage import F_xi  # noqa: E402

GAME = dict(w_h=6.0, w_l=2.0, k=1.0 / 3500.0, T=3, e_min=0.0, e_max=100.0)
PLAN_TABLE = {  # plan section 1.1
    60: {"B": 220, "D2": 220, "D3": 440, "bins": (44, 88), "dev": (1, 111, 221), "final": (1, 221, 441),
         "dense": 26403},
    50: {"B": 200, "D2": 200, "D3": 400, "bins": (40, 80), "dev": (1, 101, 201), "final": (1, 201, 401),
         "dense": 24003},
}


def spec_q(q: float) -> GameSpec:
    return GameSpec(**GAME, q=float(q))


def new_agent(seed: int, q: int) -> Tuple[CurriculumPPO, Dict[str, np.random.Generator]]:
    m = bm.build_manifest("pilot", q, [bm.SEEDS["pilot"][q][0]])
    gen, rngs = rx.make_streams(seed, q, m["resolved"]["rng"]["namespaces"])
    return CurriculumPPO(rx.ppo_config(m["resolved"]), gen, rngs["minibatch"]), rngs


# ===========================================================================
# Task 1 — manifests
# ===========================================================================

class ManifestTests(unittest.TestCase):
    def test_numeric_tables_both_q(self) -> None:
        for q, exp in PLAN_TABLE.items():
            m = bm.build_manifest("pilot", q, bm.SEEDS["pilot"][q])
            der = m["resolved"]["derived"]
            self.assertEqual(m["resolved"]["game"]["q"], float(q))
            self.assertEqual(der["B"], exp["B"])
            self.assertEqual(der["dw"], 4.0)
            self.assertEqual(der["domains"]["2"], [-exp["D2"], exp["D2"]])
            self.assertEqual(der["domains"]["3"], [-exp["D3"], exp["D3"]])
            self.assertEqual((der["bins"]["2"], der["bins"]["3"]), exp["bins"])
            self.assertEqual(tuple(der["dev_grid_points"][t] for t in "123"), exp["dev"])
            self.assertEqual(tuple(der["final_grid_points"][t] for t in "123"), exp["final"])
            self.assertEqual(der["dense_points_total"], exp["dense"])
            spec = spec_q(q)
            dense = sum(metrics.dense_points(spec.domain_half(t), 0.05).size for t in (1, 2, 3))
            self.assertEqual(dense, exp["dense"])

    def test_per_update_counts_pilot_and_smoke(self) -> None:
        pilot = bm.build_manifest("pilot", 60, [10401])["resolved"]["phases"]
        smoke = bm.build_manifest("smoke", 60, [10400])["resolved"]["phases"]
        self.assertEqual([p["per_update"]["episodes"] for p in pilot], [512, 512, 512])
        self.assertEqual([p["per_update"]["joint_environment_steps"] for p in pilot], [512, 1024, 1109])
        self.assertEqual(pilot[2]["per_update"]["transitions_by_stage"], {"1": 256, "2": 341, "3": 512})
        self.assertEqual([p["per_update"]["episodes"] for p in smoke], [32, 32, 32])
        self.assertEqual(smoke[2]["start_counts"], {"1": 16, "2": 5, "3": 11})
        self.assertEqual(smoke[2]["per_update"]["joint_environment_steps"], 69)
        total = sum(p["cap"] * p["per_update"]["episodes"] for p in pilot)
        steps = sum(p["cap"] * p["per_update"]["joint_environment_steps"] for p in pilot)
        self.assertEqual((total, steps, 2 * steps), (1433600, 2815400, 5630800))
        exp = {60: (2327.2727, 1.931818, 1.943182), 50: (2560.0, 2.125, 2.1375)}
        for q, (a, c2, c3) in exp.items():
            ph = bm.build_manifest("pilot", q, bm.SEEDS["pilot"][q])["resolved"]["phases"]
            self.assertAlmostEqual(ph[0]["expected_direct_es_per_bin_per_update"]["3"] * 400, a, places=3)
            self.assertAlmostEqual(ph[2]["expected_direct_es_per_bin_per_update"]["2"], c2, places=5)
            self.assertAlmostEqual(ph[2]["expected_direct_es_per_bin_per_update"]["3"], c3, places=5)

    def test_phase_dynamics_settings(self) -> None:
        pilot = {p["name"]: p for p in bm.build_manifest("pilot", 50, [10411])["resolved"]["phases"]}
        self.assertEqual([pilot[n]["cap"] for n in "ABC"], [400, 600, 1800])
        self.assertEqual([pilot[n]["check_every"] for n in "ABC"], [100, 25, 25])
        self.assertEqual((pilot["A"]["k_consecutive"], pilot["B"]["k_consecutive"], pilot["C"]["k_consecutive"]),
                         (None, 3, 1))
        self.assertEqual(pilot["B"]["concentration_stages"], [2, 3])
        self.assertEqual(pilot["C"]["concentration_stages"], [1, 2, 3])
        smoke = {p["name"]: p for p in bm.build_manifest("smoke", 50, [10410])["resolved"]["phases"]}
        self.assertEqual([smoke[n]["cap"] for n in "ABC"], [2, 2, 2])
        self.assertEqual([smoke[n]["check_every"] for n in "ABC"], [1, 1, 1])
        self.assertEqual((smoke["B"]["k_consecutive"], smoke["C"]["k_consecutive"]), (3, 1))

    def test_smoke_overrides_do_not_leak(self) -> None:
        before = copy.deepcopy(bm.PHASES)
        bm.build_manifest("smoke", 60, [10400])
        self.assertEqual(bm.PHASES, before)
        pm = bm.build_manifest("pilot", 60, [10401])
        self.assertEqual(pm["resolved"]["smoke_overrides"], {})
        self.assertEqual(pm["resolved"]["economics"]["replicates"], 3)
        self.assertEqual(pm["resolved"]["economics"]["episodes_per_replicate"], 200000)
        sm = bm.build_manifest("smoke", 60, [10400])
        self.assertEqual(sm["resolved"]["economics"]["episodes_per_replicate"], 200)
        self.assertEqual(sm["resolved"]["ppo"], pm["resolved"]["ppo"])
        self.assertEqual(sm["resolved"]["verifier"], pm["resolved"]["verifier"])

    def test_ppo_values_are_existing_defaults(self) -> None:
        d = asdict(PPOConfig())
        d["adam_betas"] = list(d["adam_betas"])
        self.assertEqual(bm.PPO, d)

    def test_seed_exclusion(self) -> None:
        with self.assertRaises(ValueError):
            bm.build_manifest("pilot", 60, [10400])
        with self.assertRaises(ValueError):
            bm.build_manifest("pilot", 50, [10301])
        with self.assertRaises(ValueError):
            bm.build_manifest("smoke", 60, [11001])
        fs = {"resolved_non_q": {k: v for k, v in bm.resolved_config(60, "pilot").items()
                                 if k not in ("game", "derived")}, "seeds_per_q": 10}
        with self.assertRaises(ValueError):
            bm.build_manifest("formal", 60, [10401], fs)
        with self.assertRaises(ValueError):
            bm.build_manifest("pilot", 60, [10401, 10401])

    def test_resolved_q_and_unique_paths(self) -> None:
        ids, dirs = set(), set()
        fs = {"resolved_non_q": {k: v for k, v in bm.resolved_config(60, "pilot").items()
                                 if k not in ("game", "derived")}, "seeds_per_q": 20}
        for cohort in ("smoke", "pilot", "formal"):
            for q in (60, 50):
                m = bm.build_manifest(cohort, q, bm.SEEDS[cohort][q], fs if cohort == "formal" else None)
                self.assertEqual(m["resolved"]["game"]["q"], float(q))
                self.assertEqual(GameSpec(**m["resolved"]["game"]).B, PLAN_TABLE[q]["B"])
                for r in m["runs"]:
                    self.assertNotIn(r["run_id"], ids)
                    self.assertNotIn(r["output_dir"], dirs)
                    ids.add(r["run_id"])
                    dirs.add(r["output_dir"])
                    self.assertIn(f"_q{q}_s{r['seed']}", r["run_id"])

    def test_formal_branch_same_settings_no_smoke(self) -> None:
        fs = {"resolved_non_q": {k: v for k, v in bm.resolved_config(50, "pilot").items()
                                 if k not in ("game", "derived")}, "seeds_per_q": 10}
        m50 = bm.build_manifest("formal", 50, bm.SEEDS["formal"][50][:10], fs)
        m60 = bm.build_manifest("formal", 60, bm.SEEDS["formal"][60][:10], fs)
        strip = lambda ph: [{k: v for k, v in p.items() if k != "expected_direct_es_per_bin_per_update"} for p in ph]
        self.assertEqual(strip(m50["resolved"]["phases"]), strip(m60["resolved"]["phases"]))
        self.assertEqual(m50["seeds"], list(range(11001, 11011)))
        self.assertEqual(m50["resolved"]["verifier"], bm.resolved_config(50, "pilot")["verifier"])
        alt = copy.deepcopy(fs)
        alt["resolved_non_q"]["verifier"]["final"]["gl_half"] = 48          # every setting is applied
        self.assertEqual(bm.build_manifest("formal", 50, [11001], alt)["resolved"]["verifier"]["final"]["gl_half"], 48)
        bad = copy.deepcopy(fs)
        bad["resolved_non_q"]["smoke_overrides"] = bm.SMOKE_OVERRIDES
        with self.assertRaises(ValueError):
            bm.build_manifest("formal", 50, [11001], bad)
        bad = copy.deepcopy(fs)
        bad["resolved_non_q"]["game"] = {"q": 50}
        with self.assertRaises(ValueError):
            bm.build_manifest("formal", 50, [11001], bad)

    def test_no_legacy_fields(self) -> None:
        text = json.dumps(bm.build_manifest("pilot", 60, [10401]))
        for legacy in ('"warmup"', '"stability_every"', '"B1"', '"k_stop"', '"verifier_timeout"'):
            self.assertNotIn(legacy, text)


# ===========================================================================
# Task 2 — collector and coverage
# ===========================================================================

class StubAgent:
    """Deterministic learner 50; opponent 90 if its own observed gap > 0 else 10."""

    opponent = "opponent"

    def beta_params(self, obs: np.ndarray, net: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        d = obs[:, 1].astype(float)
        mean = np.full(d.size, 0.5) if net is None else np.where(d > 0, 0.9, 0.1)
        return (mean * 1000).astype(np.float32), ((1 - mean) * 1000).astype(np.float32)

    def sample_actions(self, a: np.ndarray, b: np.ndarray, rng: Any) -> np.ndarray:
        return (a / (a + b)).astype(np.float32)

    def log_prob(self, a: np.ndarray, b: np.ndarray, act: np.ndarray) -> np.ndarray:
        return np.zeros(act.size, dtype=np.float32)

    def value(self, obs: np.ndarray) -> np.ndarray:
        return (7.0 + 3.0 * obs[:, 1]).astype(np.float32)


class CollectorTests(unittest.TestCase):
    def _starts(self, spec: GameSpec, counts: Dict[str, int], seed: int) -> Tuple[np.ndarray, ...]:
        return rx.sample_starts(spec, StartSampler(spec, 10.0), counts, np.random.default_rng(seed))

    def test_matches_w_collector_and_rng_states(self) -> None:
        spec = spec_q(60)
        a1, r1 = new_agent(10401, 60)
        a2, r2 = new_agent(10401, 60)
        for counts, n_tr in (({"3": 512}, 512), ({"2": 512}, 1024), ({"1": 256, "2": 85, "3": 171}, 1109)):
            t0, d0, roles = self._starts(spec, counts, 5)
            new = collect_t3.collect_batch(spec, a1, t0, d0, roles, r1["env_noise"], r1["learner_action"],
                                           r1["opponent_action"], 1.0, 1.0, 10.0)
            old = w_collect_batch(spec, a2, t0, d0, roles, r2["env_noise"], r2["learner_action"],
                                  r2["opponent_action"], 1.0, 1.0, 10.0)
            for key in ("states", "actions", "logp", "returns", "advantages"):
                np.testing.assert_array_equal(new[key], old[key])
            for key in ("n_episodes", "n_transitions", "mean_episode_return", "mean_effort_by_stage"):
                self.assertEqual(new[key], old[key])
            self.assertEqual(new["visitation"].keys(), old["visitation"].keys())
            for name in ("env_noise", "learner_action", "opponent_action"):
                self.assertEqual(r1[name].bit_generator.state, r2[name].bit_generator.state)
            self.assertEqual(new["n_transitions"], n_tr)
            total = sum(int(v.sum()) for v in new["visitation_by_start_stage"].values())
            self.assertEqual(total, n_tr)
            self.assertEqual(sum(new["stage_transition_counts"].values()), n_tr)
            for s, n in counts.items():
                for t in range(int(s), 4):
                    self.assertEqual(int(new["visitation_by_start_stage"][f"s{s}_t{t}"].sum()), n)
                self.assertEqual(int(new["start_bins_raw"][f"s{s}"].sum()), n)
                self.assertEqual(int(new["start_bins_learner"][f"s{s}"].sum()), n)
            a1.update(new["states"], new["actions"], new["logp"], new["returns"], new["advantages"])
            a2.update(old["states"], old["actions"], old["logp"], old["returns"], old["advantages"])
        self.assertIn("s2_t3", new["visitation_by_start_stage"])

    def test_learner_signed_vs_raw_start_counts(self) -> None:
        spec = spec_q(50)
        t0 = np.full(4, 3)
        d0 = np.array([-395.0, -5.0, 5.0, 395.0])
        roles = np.array([0, 1, 0, 1])
        agent, r = new_agent(1, 50)
        out = collect_t3.collect_batch(spec, agent, t0, d0, roles, r["env_noise"], r["learner_action"],
                                       r["opponent_action"], 1.0, 1.0, 10.0)
        raw, lrn = out["start_bins_raw"]["s3"], out["start_bins_learner"]["s3"]
        self.assertEqual((raw[0], raw[39], raw[40], raw[79]), (1, 1, 1, 1))
        self.assertEqual((lrn[0], lrn[40], lrn[79]), (2, 2, 0))

    def test_semantics_with_stub_agent(self) -> None:
        spec = spec_q(50)
        n = 64
        t0 = np.array([1] * 20 + [2] * 22 + [3] * 22)
        rng0 = np.random.default_rng(3)
        d0 = np.concatenate([np.zeros(20), rng0.uniform(-200, 200, 22), rng0.uniform(-400, 400, 22)])
        roles = np.arange(n) % 2
        env = np.random.default_rng(11)
        out = collect_t3.collect_batch(spec, StubAgent(), t0, d0, roles, env, None, None, 1.0, 1.0, 10.0,
                                       trace=True)
        tr = out["trace"]
        replay = np.random.default_rng(11)
        for t in (1, 2, 3):
            s = tr["stages"][t]
            eps0 = replay.uniform(-spec.q, spec.q, size=s["idx"].size)
            eps1 = replay.uniform(-spec.q, spec.q, size=s["idx"].size)
            np.testing.assert_array_equal(s["eps0"], eps0)
            np.testing.assert_array_equal(s["eps1"], eps1)
            r = roles[s["idx"]]
            np.testing.assert_array_equal(s["eps_L"], np.where(r == 0, eps0, eps1))
            np.testing.assert_array_equal(s["eps_O"], np.where(r == 0, eps1, eps0))
            dL = tr["d_pre"][s["idx"], t]
            np.testing.assert_array_equal(s["obs_O"][:, 1], -s["obs_L"][:, 1])
            e_O_expected = np.where(-dL > 0, 90.0, 10.0)                     # opponent observes -d
            np.testing.assert_allclose(s["e_O"], e_O_expected, atol=1e-4)
            np.testing.assert_allclose(s["d_next"], dL + s["e_L"] - s["e_O"] + s["eps_L"] - s["eps_O"], atol=1e-12)
            # physical player-0 gap evolves with eps0 - eps1 regardless of roles
            sign = np.where(r == 0, 1.0, -1.0)
            e0 = np.where(r == 0, s["e_L"], s["e_O"])
            e1 = np.where(r == 0, s["e_O"], s["e_L"])
            np.testing.assert_allclose(sign * s["d_next"], sign * dL + e0 - e1 + eps0 - eps1, atol=1e-12)
        rew, ret, val = tr["rewards"], tr["ret"], tr["values"]
        self.assertTrue(np.all(val[:, 3] != 0))
        np.testing.assert_array_equal(ret[:, 3], rew[:, 3])                   # zero terminal bootstrap
        es2 = np.nonzero(t0 == 2)[0]
        np.testing.assert_allclose(ret[es2, 2], rew[es2, 2] + rew[es2, 3], atol=1e-12)
        self.assertTrue(np.all(np.isnan(tr["d_pre"][es2, 1])))
        self.assertTrue(np.all(np.isfinite(tr["d_pre"][es2, 3])))            # ES2 continues to stage 3
        e3 = tr["stages"][3]["e_L"]
        prize = rew[tr["stages"][3]["idx"], 3] + spec.k * e3 ** 2
        dfin = tr["d_final"][tr["stages"][3]["idx"]]
        np.testing.assert_allclose(prize, np.where(dfin > 0, 6.0, np.where(dfin < 0, 2.0, 4.0)), atol=1e-12)
        np.testing.assert_allclose(rew[es2, 2], -spec.k * tr["stages"][2]["e_L"][np.isin(tr["stages"][2]["idx"], es2)] ** 2)

    def test_strict_bins_boundaries(self) -> None:
        spec = spec_q(60)
        half = spec.domain_half(3)
        eps = 1e-9 * half
        b = metrics.strict_gap_bins(spec, 3, np.array([-half, 0.0, half, -half - 0.4 * eps, half + 0.4 * eps]), 10.0)
        self.assertEqual(b.tolist(), [0, 44, 87, 0, 87])
        for bad in (half + 2 * eps, -half - 1e-3, np.nan):
            with self.assertRaises(metrics.CoverageDomainError):
                metrics.strict_gap_bins(spec, 3, np.array([bad]), 10.0, origin="test")
        self.assertEqual(metrics.strict_gap_bins(spec, 1, np.array([0.0, 1e-12]), 10.0).tolist(), [0, 0])
        with self.assertRaises(metrics.CoverageDomainError):
            metrics.strict_gap_bins(spec, 1, np.array([1e-3]), 10.0)

    def test_root_starts_never_use_interval_sampler(self) -> None:
        spec = spec_q(60)
        sampler = StartSampler(spec, 10.0)
        with patch.object(StartSampler, "balanced", side_effect=AssertionError("ES sampler called")):
            t0, d0, _ = rx.sample_starts(spec, sampler, {"1": 16}, np.random.default_rng(0))
        self.assertTrue(np.all(d0 == 0) and np.all(t0 == 1))


# ===========================================================================
# Task 3 — state machine
# ===========================================================================

def scripted(phases: List[Dict[str, Any]], outcomes: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Run the state machine with scripted check outcomes (True/False/'invalid')."""
    log: Dict[str, Any] = {"records": [], "minimum": [], "candidate": [], "train": 0}
    it = {k: iter(outcomes.get(k, [])) for k in ("A", "B", "C")}

    def check(ph: Dict[str, Any], local: int, g: int) -> Tuple[Dict[str, Any], Any]:
        o = next(it[ph["name"]], False)
        valid = o != "invalid"
        rec = {"phase": ph["name"], "local_update": local, "global_update": g, "valid": valid,
               "eligible": None if ph["name"] == "A" else bool(o is True),
               "criterion_value_over_dw": 0.0 if o is True else (None if not valid else 1.0)}
        return rec, ("res" if valid else None)

    def update_minimum(rec: Dict[str, Any], res: Any) -> None:
        if rec["valid"]:
            log["minimum"].append(rec["global_update"])

    def train(ph: Dict[str, Any], local: int, g: int) -> None:
        log["train"] += 1

    out = rx.run_phases(phases, train_step=train, check=check,
                        on_phase_entry=lambda ph, g: None, on_phase_exit=lambda row: None,
                        log_verifier=lambda r: log["records"].append(r), update_minimum=update_minimum,
                        save_candidate=lambda r: log["candidate"].append(r["global_update"]))
    out["log"] = log
    return out


class StateMachineTests(unittest.TestCase):
    phases = bm.resolved_config(60, "pilot")["phases"]

    def test_A_favorable_never_exits_early(self) -> None:
        out = scripted(self.phases, {"A": [True] * 4, "B": [True] * 3, "C": [True]})
        self.assertEqual(out["phases"][0]["local_updates"], 400)
        self.assertEqual(out["phases"][0]["exit_reason"], "fixed_budget_completed")
        self.assertEqual([r["consecutive_eligible"] for r in out["log"]["records"][:4]], [0, 0, 0, 0])

    def test_B_sequence_exits_at_150(self) -> None:
        out = scripted(self.phases, {"B": [True, True, False, True, True, True], "C": [True]})
        b = out["phases"][1]
        self.assertEqual((b["local_updates"], b["exit_reason"]), (150, "verifier_passed"))
        self.assertEqual([r["consecutive_eligible"] for r in out["log"]["records"] if r["phase"] == "B"],
                         [1, 2, 0, 1, 2, 3])

    def test_invalid_resets_consecutive_and_B_forced(self) -> None:
        seq = [True, True, "invalid"] * 8
        out = scripted(self.phases, {"B": seq, "C": [False] * 72})
        recs = [r for r in out["log"]["records"] if r["phase"] == "B"]
        self.assertEqual([r["consecutive_eligible"] for r in recs[:6]], [1, 2, 0, 1, 2, 0])
        self.assertEqual((out["phases"][1]["local_updates"], out["phases"][1]["exit_reason"]), (600, "budget_forced"))
        self.assertEqual(out["phases"][2]["exit_reason"], "no_candidate_budget_exhausted")
        self.assertEqual(out["phases"][2]["local_updates"], 1800)
        self.assertFalse(out["has_candidate"])
        self.assertEqual(out["stop_global_update"], 2800)
        self.assertEqual(out["log"]["train"], 2800)

    def test_C_stops_at_first_pass_and_minimum_includes_candidate_call(self) -> None:
        out = scripted(self.phases, {"B": [True] * 3, "C": [False, True, True]})
        c = out["phases"][2]
        self.assertEqual((c["local_updates"], c["exit_reason"]), (50, "candidate_found"))
        self.assertTrue(out["has_candidate"])
        self.assertEqual(out["candidate_update"], 400 + 75 + 50)
        self.assertEqual(out["log"]["candidate"], [525])
        self.assertEqual(out["log"]["minimum"], [500, 525])

    def test_earliest_exits_B75_C25(self) -> None:
        out = scripted(self.phases, {"B": [True] * 3, "C": [True]})
        self.assertEqual(out["phases"][1]["local_updates"], 75)
        self.assertEqual(out["phases"][2]["local_updates"], 25)
        self.assertEqual(out["stop_global_update"], 500)

    def test_all_invalid_C_has_no_minimum(self) -> None:
        out = scripted(self.phases, {"C": ["invalid"] * 72})
        self.assertEqual(out["log"]["minimum"], [])
        self.assertEqual(metrics.diagnose_minimum_C(out["log"]["records"], None, True)["reason"],
                         "no_valid_C_verifier")

    def test_smoke_B_budget_forced(self) -> None:
        sm = bm.resolved_config(60, "smoke")["phases"]
        out = scripted(sm, {"A": [True, True], "B": [True, True], "C": [False, False]})
        self.assertEqual([p["local_updates"] for p in out["phases"]], [2, 2, 2])
        self.assertEqual(out["phases"][1]["exit_reason"], "budget_forced")


# ===========================================================================
# Task 4 — T=3 DP-BR verifier (unchanged core)
# ===========================================================================

def F_analytic(y: float, q: float) -> float:
    """CDF of eps_i - eps_j, eps ~ U(-q, q), written out piecewise."""
    if y <= -2 * q:
        return 0.0
    if y <= 0:
        return (y + 2 * q) ** 2 / (8 * q * q)
    if y < 2 * q:
        return 1.0 - (2 * q - y) ** 2 / (8 * q * q)
    return 1.0


def interp_scalar(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    for j in range(len(xs) - 1):
        if xs[j] - 1e-12 <= x <= xs[j + 1] + 1e-12:
            w = (x - xs[j]) / (xs[j + 1] - xs[j])
            return (1 - w) * ys[j] + w * ys[j + 1]
    raise ValueError(f"{x} outside grid")


class VerifierTests(unittest.TestCase):
    def test_stage_order_and_all_stage_results_exported(self) -> None:
        calls: List[int] = []

        def pol(t: int, d: np.ndarray) -> np.ndarray:
            calls.append(t)
            return np.full(np.asarray(d).shape, 50.0)

        res = dpv.verify(pol, w_h=6, w_l=2, k=1 / 3500, q=50, T=3, cfg=dpv.DEV_CONFIG)
        self.assertEqual(calls, [3, 3, 2, 2, 1, 1])
        self.assertEqual(sorted(res.stages), [1, 2, 3])
        arrays = dpv.stage_result_arrays(res, "x")
        for t in (1, 2, 3):
            self.assertIn(f"x_t{t}_v_br", arrays)
        self.assertEqual(res.v_br_root, res.stages[1].v_br[0])

    def test_continuation_gain_differs_from_one_step_delta(self) -> None:
        spec = spec_q(50)
        res = dpv.verify(lambda t, d: np.full(np.asarray(d).shape, 50.0), w_h=6, w_l=2, k=spec.k, q=50, T=3,
                         cfg=dpv.DEV_CONFIG)
        ph = bm.resolved_config(50, "pilot")["phases"][1]
        conc = {"valid": True, "max_std_norm": 0.01}
        rec = rx.development_record(ph, res, None, conc, 0.04, 4.0)
        cont = float(np.max(res.stages[2].v_br - res.stages[2].v_mean))
        self.assertAlmostEqual(rec["criterion_value_raw"], cont)
        self.assertGreater(rec["criterion_value_over_dw"] - res.full_delta_max[2] / 4.0, 0.01)
        self.assertEqual(rec["criterion"], "max_D2_continuation_gain_over_dw")

    def test_constant_equal_effort_mean_recursion(self) -> None:
        k, e = 1 / 3500, 37.5
        for q in (50, 60):
            res = dpv.verify(lambda t, d: np.full(np.asarray(d).shape, e), w_h=6, w_l=2, k=k, q=q, T=3,
                             cfg=dpv.FINAL_CONFIG)
            self.assertLess(abs(res.v_mean_root - (4.0 - 3 * k * e * e)), 1e-8)
            g3 = res.stages[3].d_grid
            exp3 = np.array([-k * e * e + 2 + 4 * F_analytic(x, q) for x in g3])
            np.testing.assert_allclose(res.stages[3].v_mean, exp3, atol=1e-12)
            v2 = res.stages[2].v_mean
            np.testing.assert_allclose(v2 + v2[::-1], 2 * (4 - 2 * k * e * e), atol=1e-9)

    def test_scalar_loop_oracle_small_fixture(self) -> None:
        q, wh, wl, k, emin, emax = 2.0, 6.0, 2.0, 10.0, 1.0, 3.0
        cfg = dpv.VerifierConfig("oracle", state_step=1.0, effort_step=0.5, gl_half=4)
        res = dpv.verify(lambda t, d: np.full(np.asarray(d).shape, 1.0), w_h=wh, w_l=wl, k=k, q=q, T=3,
                         e_min=emin, e_max=emax, cfg=cfg)
        B = (emax - emin) + 2 * q
        grids = {1: np.zeros(1), 2: np.linspace(-B, B, 13), 3: np.linspace(-2 * B, 2 * B, 25)}
        efforts = [1.0, 1.5, 2.0, 2.5, 3.0]
        x, w = np.polynomial.legendre.leggauss(4)
        nodes, weights = [], []
        for xi, wi in zip(x, w):
            for c in (-q, q):
                node = c + q * xi
                nodes.append(node)
                weights.append(q * wi * (2 * q - abs(node)) / (4 * q * q))
        self.assertAlmostEqual(sum(weights), 1.0, places=14)
        V: Dict[int, np.ndarray] = {}
        for t in (3, 2, 1):
            vals, args_ = [], []
            for d in grids[t]:
                best, arg = -math.inf, None
                for e in efforts:
                    y = d - 1.0 + e
                    if t == 3:
                        cont = wl + (wh - wl) * F_analytic(y, q)
                    else:
                        cont = sum(wt * interp_scalar(grids[t + 1], V[t + 1], y + nd) for nd, wt in zip(nodes, weights))
                    qv = -k * e * e + cont
                    if qv > best:
                        best, arg = qv, e
                vals.append(best)
                args_.append(arg)
            V[t] = np.array(vals)
            self.assertTrue(all(a == 1.0 for a in args_))
            np.testing.assert_allclose(res.stages[t].v_br, V[t], atol=1e-8, rtol=0)
            np.testing.assert_allclose(res.stages[t].a_br, 1.0, atol=1e-10, rtol=0)
            np.testing.assert_allclose(res.stages[t].v_mean, V[t], atol=1e-8, rtol=0)

    def test_terminal_expectation_closed_form_not_interpolated(self) -> None:
        for q in (50.0, 60.0):
            ys = np.linspace(-3 * q, 3 * q, 601)
            np.testing.assert_allclose(F_xi(ys, q), [F_analytic(y, q) for y in ys], atol=1e-15)
        res = dpv.verify(lambda t, d: np.full(np.asarray(d).shape, 0.0), w_h=6, w_l=2, k=1 / 3500, q=60, T=3,
                         cfg=dpv.DEV_CONFIG)
        g3 = res.stages[3].d_grid
        np.testing.assert_allclose(res.stages[3].v_mean, [2 + 4 * F_analytic(x, 60.0) for x in g3], atol=1e-12)
        step = np.where(g3 > 0, 6.0, np.where(g3 < 0, 2.0, 4.0))
        self.assertGreater(np.max(np.abs(res.stages[3].v_mean - step)), 0.5)

    def test_br_reach_forward_union_and_pmf_mass(self) -> None:
        spec = spec_q(50)
        res = dpv.verify(lambda t, d: 30.0 + 0.05 * np.asarray(d), w_h=6, w_l=2, k=spec.k, q=50, T=3,
                         cfg=dpv.DEV_CONFIG)
        q, tol = 50.0, dpv.DEV_CONFIG.grid_tol
        reach = {1: np.array([True])}
        for t in (1, 2):
            s = res.stages[t]
            nxt = res.stages[t + 1].d_grid
            lo_dom, hi_dom = nxt[0], nxt[-1]
            mask = np.zeros(nxt.size, dtype=bool)
            for i in np.nonzero(reach[t])[0]:
                c = s.d_grid[i] + s.a_br[i] - s.e_opp[i]
                lo, hi = max(c - 2 * q, lo_dom), min(c + 2 * q, hi_dom)
                for j, g in enumerate(nxt):
                    if lo - tol <= g <= hi + tol:
                        mask[j] = True
            reach[t + 1] = mask
            np.testing.assert_array_equal(res.stages[t + 1].reach, mask)
        self.assertLess(reach[2].sum(), reach[2].size)
        for t in (1, 2, 3):
            self.assertLess(abs(res.stages[t].pmf.sum() - 1.0), 1e-10)

    def test_domain_error_and_roundoff_tolerance(self) -> None:
        q, emin, emax = 2.0, 1.0, 3.0
        cfg = dpv.VerifierConfig("oracle", state_step=1.0, effort_step=0.5, gl_half=4)
        x_max = q + q * np.polynomial.legendre.leggauss(4)[0].max()
        land_max = 6.0 - 1.0 + emax + x_max
        orig = dpv.stage_grid

        def shrunk(H: float):
            def g(t: int, B: float, step: float) -> np.ndarray:
                out = orig(t, B, step)
                return out * (H / out[-1]) if t == 3 else out
            return g

        pol = lambda t, d: np.full(np.asarray(d).shape, 1.0)
        kw = dict(w_h=6, w_l=2, k=10, q=q, T=3, e_min=emin, e_max=emax, cfg=cfg)
        eps = cfg.grid_tol * land_max
        with patch.object(dpv, "stage_grid", shrunk(11.0)):
            with self.assertRaises(dpv.DomainError):
                dpv.verify(pol, **kw)
        with patch.object(dpv, "stage_grid", shrunk(land_max - 0.4 * eps)):
            self.assertTrue(dpv.verify(pol, **kw).valid)
        with patch.object(dpv, "stage_grid", shrunk(land_max - 3 * eps)):
            with self.assertRaises(dpv.DomainError):
                dpv.verify(pol, **kw)
        spec = GameSpec(w_h=6, w_l=2, k=10, q=q, T=3, e_min=emin, e_max=emax)
        with patch.object(dpv, "stage_grid", shrunk(11.0)):
            res, err = run_verifier(pol, None, spec, cfg)
        self.assertIsNone(res)
        rec = rx.development_record(bm.PHASES[2], res, err, {"valid": True, "max_std_norm": 0.0}, 0.04, 4.0)
        self.assertFalse(rec["valid"])
        self.assertFalse(rec["eligible"])
        self.assertIn("DomainError", rec["invalid_reasons"][0])

    def test_invalid_reasons_reach_the_record(self) -> None:
        pol = lambda t, d: np.full(np.asarray(d).shape, 40.0)
        for field, key in (("mass_tol", "pmf mass"), ("pdl_tol_over_dw", "PDL residual")):
            cfg = dpv.VerifierConfig("dev_bad", 4.0, 1.0, 16, **{field: -1.0})
            res = dpv.verify(pol, w_h=6, w_l=2, k=1 / 3500, q=60, T=3, cfg=cfg)
            self.assertFalse(res.valid)
            rec = rx.development_record(bm.PHASES[2], res, None, {"valid": True, "max_std_norm": 0.0}, 0.04, 4.0)
            self.assertFalse(rec["eligible"])
            self.assertEqual(rec["failure_type"], "invalid")
            self.assertTrue(any(key in r for r in rec["invalid_reasons"]))
            self.assertIsNotNone(rec["summary"]["dreach"])       # finite diagnostics kept

    def test_real_dev_and_final_tiers_q50_q60(self) -> None:
        for q, exp in PLAN_TABLE.items():
            spec = spec_q(q)
            agent, _ = new_agent(123, q)
            mean_fn, beta_fn = make_policy_fns(agent, spec)
            for cfg, pts in ((dpv.DEV_CONFIG, exp["dev"]), (dpv.FINAL_CONFIG, exp["final"])):
                res, err = run_verifier(mean_fn, beta_fn, spec, cfg)
                self.assertIsNone(err)
                s = res.summary()
                self.assertEqual(tuple(s["grid_points"][t] for t in "123"), pts)
                for key in ("exp_root", "dreach", "dfull", "delta_max_all", "pdl_residual"):
                    self.assertTrue(math.isfinite(s[key]))
            prof = metrics.policy_profiles(mean_fn, beta_fn, spec, 0.05)
            self.assertEqual(prof["concentration"]["n_points"], exp["dense"])
            self.assertEqual(prof["mean_fn_consistency_max_abs"], 0.0)
            self.assertEqual(prof["arrays"]["t3_d"][0], -exp["D3"])


# ===========================================================================
# Task 5 — endpoint evaluation and certification
# ===========================================================================

def good_tiers(dreach: float = 0.02, exp_: float = 0.01) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = {"valid": True, "dreach": dreach, "exp_root": exp_, "delta_max_all": 0.1, "dfull": 0.2,
            "reach_delta_max": {"1": 0.01, "2": 0.005, "3": 0.005}, "full_delta_max": {"1": 0.01, "2": 0.05, "3": 0.1}}
    return copy.deepcopy(base), copy.deepcopy(base)


class CertificationTests(unittest.TestCase):
    cert = bm.resolved_config(60, "pilot")["final_certification"]
    conc_ok = {"valid": True, "max_std_norm": 0.03}

    def test_terminal_numeric_pass_is_not_joint_pass(self) -> None:
        dev, fin = good_tiers()
        f = metrics.certification_flags(dev, fin, self.conc_ok, False, self.cert, 4.0)
        self.assertTrue(f["numeric_thresholds_pass"])
        self.assertFalse(f["final_joint_pass"])
        self.assertEqual(f["certification"], "not_applicable_no_candidate")
        f = metrics.certification_flags(dev, fin, self.conc_ok, True, self.cert, 4.0)
        self.assertEqual((f["final_joint_pass"], f["certification"]), (True, "certified"))

    def test_invalid_final(self) -> None:
        dev, fin = good_tiers()
        fin["valid"] = False
        f = metrics.certification_flags(dev, fin, self.conc_ok, True, self.cert, 4.0)
        self.assertEqual((f["valid_final"], f["main_pass"], f["certification"]), (False, False, "not_certified"))
        f = metrics.certification_flags(dev, None, self.conc_ok, True, self.cert, 4.0)
        self.assertFalse(f["numeric_thresholds_pass"])
        self.assertIsNone(f["refine_dreach_diff_over_dw"])

    def test_refine_root_fail(self) -> None:
        dev, fin = good_tiers()
        fin["exp_root"] = dev["exp_root"] + 0.0021 * 4
        f = metrics.certification_flags(dev, fin, self.conc_ok, True, self.cert, 4.0)
        self.assertFalse(f["refine_exp_pass"])
        self.assertTrue(f["refine_dreach_pass"] and f["main_pass"])
        self.assertEqual(f["certification"], "not_certified")

    def test_refine_reach_and_main_fail(self) -> None:
        dev, fin = good_tiers()
        fin["dreach"] = dev["dreach"] + 0.0021 * 4
        f = metrics.certification_flags(dev, fin, self.conc_ok, True, self.cert, 4.0)
        self.assertFalse(f["refine_dreach_pass"])
        dev, fin = good_tiers(dreach=0.0401)
        f = metrics.certification_flags(dev, fin, self.conc_ok, True, self.cert, 4.0)
        self.assertFalse(f["main_pass"])
        self.assertTrue(f["refine_dreach_pass"])

    def test_dense_concentration_fail(self) -> None:
        dev, fin = good_tiers()
        for conc in ({"valid": True, "max_std_norm": 0.0401}, {"valid": False, "max_std_norm": 0.01}, {}):
            f = metrics.certification_flags(dev, fin, conc, True, self.cert, 4.0)
            self.assertFalse(f["dense_conc_pass"])
            self.assertFalse(f["final_joint_pass"])

    def test_evaluate_checkpoint_reads_saved_file(self) -> None:
        cfg = bm.resolved_config(60, "smoke")
        spec = GameSpec(**cfg["game"])
        agent, _ = new_agent(77, 60)
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "checkpoints"
            ident = {"checkpoint_kind": "candidate", "global_update": 5}
            files = rx.save_checkpoint_files(agent, ck, "endpoint", ident)
            path = Path(tmp) / files["pt"]
            mtime = path.stat().st_mtime_ns
            out, arrays, rel = rx.evaluate_checkpoint(str(path), cfg, "candidate")
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            self.assertEqual(out["checkpoint_identity"], ident)
            for k, v in agent.actor.state_dict().items():
                self.assertTrue(torch.equal(v, rel.actor.state_dict()[k]))
            mean_fn, beta_fn = make_policy_fns(agent, spec)
            res, _ = run_verifier(mean_fn, beta_fn, spec, dpv.DEV_CONFIG)
            self.assertEqual(out["tiers"]["development"]["dreach"], res.dreach)
            self.assertEqual(out["dense_points_by_stage"], {"1": 1, "2": 8801, "3": 17601})
            self.assertIn("final_t3_pmf", arrays)
            self.assertEqual(set(out["pass_flags"]) >= {"main_pass", "refine_dreach_pass", "refine_exp_pass",
                                                        "dense_conc_pass", "numeric_thresholds_pass"}, True)
            w = np.load(Path(tmp) / files["weights_npz"])
            self.assertEqual(json.loads(str(w["identity_json"])), ident)


# ===========================================================================
# Task 6 — economics and profiles
# ===========================================================================

def const_beta(mean: float, c: float = 400.0):
    def f(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = np.asarray(d, dtype=float)
        return np.full(d.shape, mean * c), np.full(d.shape, (1 - mean) * c)
    return f


def noise(seed: int, q: float):
    g = np.random.default_rng(seed)
    return lambda t, n: (g.uniform(-q, q, n), g.uniform(-q, q, n))


class EconomicsTests(unittest.TestCase):
    def test_summarize_moments_rules(self) -> None:
        m = metrics.summarize_moments(0, 0.0, 0.0)
        self.assertIsNone(m["mean"])
        self.assertEqual(m["mean_reason"], "n=0")
        m = metrics.summarize_moments(1, 5.0, 25.0)
        self.assertEqual(m["mean"], 5.0)
        self.assertIsNone(m["mcse"])
        m = metrics.summarize_moments(4, 2.0, 2.0)
        self.assertAlmostEqual(m["variance"], (2 - 1) / 3)
        self.assertAlmostEqual(m["mcse"], math.sqrt(m["variance"] / 4))
        self.assertEqual(metrics.summarize_moments(3, 3 * 0.1, 3 * 0.1 ** 2)["variance"], 0.0)
        with self.assertRaises(ArithmeticError):
            metrics.summarize_moments(3, 3.0, 1.0)

    def test_constant_equal_effort_policy(self) -> None:
        spec = spec_q(50)
        acc = metrics.SelfPlayAccumulator(spec, 10.0)
        ch = metrics.simulate_chunk(spec, const_beta(0.5), 0, 5000, noise(1, 50.0), None)
        acc.add(ch)
        s = metrics.selfplay_summary(acc.stage, acc.episode, acc.gap_sign, acc.lf_ties, acc.outcome, acc.bins,
                                     acc.edges, spec.k)
        for t in ("1", "2", "3"):
            st = s["stages"][t]
            self.assertAlmostEqual(st["eff_p0"]["mean"], 50.0, places=10)
            self.assertAlmostEqual(st["E_e2_p0"], 2500.0, places=7)
            self.assertAlmostEqual(st["E_cost_p0"], spec.k * 2500.0, places=12)
            self.assertAlmostEqual(st["representative_E_e2"], 2500.0, places=7)
            self.assertEqual(st["eff_p0"]["variance"], 0.0)
            self.assertAlmostEqual(st["hist_mass_player0"], 1.0)
            if t != "1":
                self.assertEqual(st["lf_diff"]["mean"], 0.0)
                self.assertGreater(st["leader"]["n"], 0)
        self.assertIsNone(s["stages"]["1"]["leader"]["mean"])
        self.assertEqual(s["stages"]["1"]["leader_follower_ties"], 5000)
        self.assertEqual(s["stages"]["1"]["gap_quantiles_player0"], {"p05": 0.0, "p50": 0.0, "p95": 0.0})
        self.assertTrue(s["accounting_ok"])
        for t in (1, 2, 3):
            self.assertEqual(acc.bins[f"p0_count_t{t}"].sum(), 5000)

    def test_beta_moments_formula_and_sampling(self) -> None:
        a, b, n = 30.0, 70.0, 400000
        m = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        x = np.random.default_rng(0).beta(a, b, n)
        self.assertLess(abs(x.mean() - m), 5 * math.sqrt(var / n))
        e2 = (x * x).mean()
        self.assertLess(abs(e2 - (var + m * m)), 5 * (x * x).std() / math.sqrt(n))
        spec = spec_q(60)
        agent, _ = new_agent(9, 60)
        g = {1: np.random.default_rng(1), 2: np.random.default_rng(2)}
        ch = metrics.simulate_chunk(spec, const_beta(0.3, 100.0), 1, 100000, noise(3, 60.0),
                                    lambda p, t, a_, b_: agent.sample_actions(a_, b_, g[1 + p]))
        acc = metrics.SelfPlayAccumulator(spec, 10.0)
        acc.add(ch)
        s = metrics.selfplay_summary(acc.stage, acc.episode, acc.gap_sign, acc.lf_ties, acc.outcome, acc.bins,
                                     acc.edges, spec.k)["stages"]["2"]
        theo_e2 = s["theo_m2_p0"]["mean"]
        emp = s["eff_p0"]
        sd_e2 = math.sqrt(max(0.0, np.var(ch["e0"][1] ** 2)) / emp["n"])
        self.assertLess(abs(s["E_e2_p0"] - theo_e2), 5 * sd_e2)
        self.assertLess(abs(emp["mean"] - s["theo_mean_p0"]["mean"]), 5 * emp["mcse"])

    def test_leader_follower_with_asymmetric_policy(self) -> None:
        spec = spec_q(50)

        def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            d = np.asarray(d, dtype=float)
            m = np.where(d > 0, 0.7, np.where(d < 0, 0.3, 0.5))
            return m * 1e6, (1 - m) * 1e6

        ch = metrics.simulate_chunk(spec, beta_fn, 0, 20000, noise(5, 50.0), None)
        acc = metrics.SelfPlayAccumulator(spec, 10.0)
        acc.add(ch)
        s = metrics.selfplay_summary(acc.stage, acc.episode, acc.gap_sign, acc.lf_ties, acc.outcome, acc.bins,
                                     acc.edges, spec.k)
        for t in ("2", "3"):
            st = s["stages"][t]
            self.assertAlmostEqual(st["leader"]["mean"], 70.0, places=6)
            self.assertAlmostEqual(st["follower"]["mean"], 30.0, places=6)
            self.assertAlmostEqual(st["lf_diff"]["mean"], 40.0, places=6)
            self.assertEqual(st["leader"]["n"] + st["leader_follower_ties"], 20000)
            neg = ch["d_pre"][int(t) - 1] < 0
            np.testing.assert_allclose(ch["e1"][int(t) - 1][neg], 70.0, atol=1e-4)   # player 1 leads
        self.assertEqual(s["stages"]["1"]["leader"]["n"], 0)

    def test_swapped_physical_roles(self) -> None:
        spec = spec_q(60)
        rng = np.random.default_rng(8)
        n = 3000
        eps = {t: (rng.uniform(-60, 60, n), rng.uniform(-60, 60, n)) for t in (1, 2, 3)}
        acts = {(p, t): rng.uniform(0.1, 0.9, n).astype(np.float32) for p in (0, 1) for t in (1, 2, 3)}

        def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            m = 0.5 + 0.3 * np.tanh(np.asarray(d) / 50.0)
            return m * 200, (1 - m) * 200

        a = metrics.simulate_chunk(spec, beta_fn, 1, n, lambda t, k: eps[t], lambda p, t, x, y: acts[(p, t)])
        b = metrics.simulate_chunk(spec, beta_fn, 1, n, lambda t, k: eps[t][::-1],
                                   lambda p, t, x, y: acts[(1 - p, t)])
        np.testing.assert_allclose(b["d_pre"], -a["d_pre"], atol=1e-12)
        np.testing.assert_array_equal(b["e0"], a["e1"])
        np.testing.assert_array_equal(b["payoff0"], a["payoff1"])
        np.testing.assert_array_equal(b["payoff1"], a["payoff0"])
        acc_a, acc_b = metrics.SelfPlayAccumulator(spec, 10.0), metrics.SelfPlayAccumulator(spec, 10.0)
        acc_a.add(a)
        acc_b.add(b)
        np.testing.assert_allclose(acc_b.stage["eff_p0"], acc_a.stage["eff_p1"])
        np.testing.assert_allclose(acc_b.stage["gap_p0"][:, 1], -acc_a.stage["gap_p0"][:, 1], atol=1e-6)
        for t in (2, 3):
            np.testing.assert_array_equal(acc_b.bins[f"p0_count_t{t}"], acc_a.bins[f"p1_count_t{t}"])
        np.testing.assert_allclose(acc_b.stage["lf_diff"], acc_a.stage["lf_diff"])

    def test_rng_isolation_repeatability_and_saved_moments(self) -> None:
        spec = spec_q(60)
        agent, rngs = new_agent(10401, 60)
        cfg = dict(bm.ECONOMICS, replicates=2, episodes_per_replicate=300, chunk_size=128)
        before = {k: copy.deepcopy(g.bit_generator.state) for k, g in rngs.items()}
        torch_before = torch.get_rng_state().clone()
        agent.actor.train()
        j1, a1 = metrics.evaluate_economics(agent, spec, 10401, cfg, 10.0, v_mean_root_final=3.9)
        self.assertTrue(agent.actor.training)
        for k, g in rngs.items():
            self.assertEqual(g.bit_generator.state, before[k])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        j2, a2 = metrics.evaluate_economics(agent, spec, 10401, cfg, 10.0, v_mean_root_final=3.9)
        def strip(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k != "resources"}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o

        self.assertEqual(common.dumps(strip(j1)), common.dumps(strip(j2)))
        for key in a1:
            np.testing.assert_array_equal(a1[key], a2[key])
        for mode_id, mode in ((0, "mean"), (1, "stochastic")):
            X = a1[f"m{mode_id}_pooled_stage_X"]
            for t in range(3):
                self.assertEqual(metrics.summarize_moments(*X[t])["mean"],
                                 j1["modes"][mode]["pooled"]["stages"][str(t + 1)]["X"]["mean"])
            self.assertEqual(int(a1[f"m{mode_id}_pooled_stage_eff_p0"][0][0]), 600)
        self.assertEqual(j1["modes"]["mean"]["replicates"][1]["seed_sequence_base"], [9005000, 10401, 60, 100, 0, 1])
        self.assertIsNotNone(j1["mean_payoff_vs_dp"]["difference"])

    def test_policy_profiles_fields(self) -> None:
        spec = spec_q(60)

        def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            m = 0.4 + 0.1 * np.tanh(np.asarray(d, dtype=float) / 100.0)
            return m * 300, (1 - m) * 300

        mean_fn = lambda t, d: 100 * beta_fn(t, d)[0] / 300
        prof = metrics.policy_profiles(mean_fn, beta_fn, spec, 0.05)
        A = prof["arrays"]
        self.assertEqual(A["t1_d"].tolist(), [0.0])
        self.assertEqual(A["t2_d"].size, 8801)
        j = int(np.argmin(np.abs(A["t3_asym_x"] - 50.0)))
        x = A["t3_asym_x"][j]
        self.assertAlmostEqual(A["t3_lead_follow_policy_difference"][j], mean_fn(3, x) - mean_fn(3, -x))
        np.testing.assert_allclose(A["t2_std_norm"], np.sqrt(A["t2_effort_variance"]) / 100.0, rtol=1e-12)
        self.assertAlmostEqual(A["t3_d_normalized"][-1], 1.0)


# ===========================================================================
# Task 7 — reporting rules
# ===========================================================================

class ReportTests(unittest.TestCase):
    def test_wilson(self) -> None:
        w = metrics.wilson(0, 10)
        self.assertAlmostEqual(w["hi"], 0.2775, places=4)
        self.assertEqual(w["lo"], 0.0)
        self.assertIsNone(metrics.wilson(0, 0)["p"])
        w = metrics.wilson(5, 10)
        self.assertAlmostEqual(w["lo"] + w["hi"], 1.0, places=12)

    def test_aggregate_fixture_denominators(self) -> None:
        rows = [
            {"state": "done", "has_candidate": True, "certification": "certified"},          # certified
            {"state": "done", "has_candidate": True, "certification": "not_certified"},      # candidate-final-fail
            {"state": "done", "has_candidate": False, "certification": "not_applicable_no_candidate"},  # no cand
            {"state": "done", "has_candidate": False, "certification": "not_applicable_no_candidate"},  # no valid C
            {"state": "done", "has_candidate": False, "certification": "not_applicable_no_candidate"},  # B forced
            {"state": "failed", "has_candidate": True, "certification": "error"},            # op error after cand
            {"state": "failed", "has_candidate": False, "certification": None},              # op error
            {"state": "pending", "has_candidate": None, "certification": None},              # pending
        ]
        a = make_report.aggregate(rows, "debug")
        self.assertEqual((a["N_planned"], a["N_started"], a["N_completed"], a["N_operational_failed"],
                          a["N_pending"]), (8, 7, 5, 2, 1))
        p = a["primary"]
        self.assertEqual((p["candidate_discovery"]["k"], p["candidate_discovery"]["n"]), (3, 7))
        self.assertEqual((p["conditional_certification"]["k"], p["conditional_certification"]["n"]), (1, 3))
        self.assertEqual((p["end_to_end"]["k"], p["end_to_end"]["n"]), (1, 7))
        self.assertEqual(p["n_candidate_certification_error"], 1)
        c = a["completed_only_sensitivity"]
        self.assertEqual((c["candidate_discovery"]["k"], c["candidate_discovery"]["n"]), (2, 5))
        zero = make_report.aggregate([{"state": "done", "has_candidate": False, "certification": None}], "x")
        self.assertIsNone(zero["primary"]["conditional_certification"]["p"])

    def test_read_jsonl_truncated_last_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "h.jsonl"
            p.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
            recs, info = common.read_jsonl(p)
            self.assertEqual([r["a"] for r in recs], [1, 2])
            self.assertTrue(info["truncated_last_line"])
            p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
            with self.assertRaises(ValueError):
                common.read_jsonl(p)

    def test_strict_json(self) -> None:
        self.assertEqual(json.loads(common.dumps({"x": float("nan"), "y": np.float32(1.5)})), {"x": None, "y": 1.5})

    def test_diagnose_minimum(self) -> None:
        def rec(g: int, v: float, valid: bool = True, phase: str = "C") -> Dict[str, Any]:
            return {"phase": phase, "valid": valid, "global_update": g, "local_update": g - 1000, "call_index": g,
                    "concentration": {"max_std_norm": 0.03, "stage": 3, "d": 1.0},
                    "summary": {"dreach_over_dw": v, "dreach": 4 * v,
                                "reach_delta_max": {"1": 4 * v / 2, "2": 4 * v / 4, "3": 4 * v / 4},
                                "reach_argmax_d": {"1": 0.0, "2": -8.0, "3": 12.0},
                                "full_delta_max": {"1": 1.0, "2": 2.0, "3": 2.0},
                                "full_argmax_d": {"1": 0.0, "2": 4.0, "3": 8.0}}}
        self.assertEqual(metrics.diagnose_minimum_C([], None, False)["reason"], "C_not_reached")
        recs = [rec(1025, 0.05), rec(1050, 0.03), rec(1075, 0.03), rec(1100, 0.01, valid=False), rec(900, 0.001, phase="B")]
        dg = metrics.diagnose_minimum_C(recs, {"identity": {"global_update": 1050, "call_index": 1050}}, True)
        self.assertEqual(dg["min_C_global_update"], 1050)
        self.assertTrue(dg["min_record_matches_selected_call"])
        self.assertTrue(dg["contribution_sum_matches_dreach"])
        self.assertEqual(dg["max_reach_state"], {"stage": 1, "d": 0.0, "delta": 0.06})
        self.assertEqual(dg["max_all_state"]["stage"], 2)


# ===========================================================================
# Real Run object on a tiny temporary manifest
# ===========================================================================

def tmp_manifest(tmp: Path, q: int, seed: int) -> Path:
    m = bm.build_manifest("smoke", q, bm.SEEDS["smoke"][q])
    m["resolved"]["economics"].update(replicates=2, episodes_per_replicate=100, chunk_size=64)
    m["runs"] = [dict(m["runs"][0], seed=seed, run_id=f"unit_q{q}_s{seed}", output_dir=str(tmp / f"run_{seed}"))]
    p = tmp / f"m_{seed}.json"
    p.write_text(json.dumps(m))
    return p


class RunTests(unittest.TestCase):
    def test_tiny_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mp = tmp_manifest(Path(tmp), 60, 99)
            run = rx.Run(mp, 99)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.execute(), 0)
            d = run.out
            st = json.loads((d / "status.json").read_text())
            self.assertEqual(st["state"], "done")
            hist, _ = common.read_jsonl(d / "history.jsonl")
            calls, _ = common.read_jsonl(d / "verifier_calls.jsonl")
            self.assertEqual([h["n_environment_steps"] for h in hist], [32, 32, 64, 64, 69, 69][:len(hist)])
            self.assertEqual(st["last_completed_update"], len(hist))
            self.assertEqual([c["phase"] for c in calls][:4], ["A", "A", "B", "B"])
            self.assertEqual([c["consecutive_eligible"] for c in calls if c["phase"] == "A"], [0, 0])
            fe = json.loads((d / "final_eval.json").read_text())
            er = json.loads((d / "checkpoints" / "endpoint_record.json").read_text())
            self.assertEqual(fe["checkpoint_identity"], er["identity"])
            self.assertTrue(fe["reload_identity"]["identical"])
            self.assertEqual(fe["dev_recompute_vs_call"]["dreach_over_dw"], 0.0)
            if not fe["has_candidate"]:
                self.assertEqual(fe["certification"], "not_applicable_no_candidate")
                self.assertFalse(fe["final_joint_pass"])
            cov = np.load(d / "coverage.npz")
            self.assertEqual(cov["snapshot_counts"].shape[0], len(calls))
            mr = d / "checkpoints" / "min_dev_record.json"
            if mr.exists():
                meta = json.loads(mr.read_text())
                prof = json.loads((d / "checkpoints" / "min_dev_profiles.json").read_text())
                self.assertEqual(prof["dev_replay_abs_diff"], 0.0)
                dg = metrics.diagnose_minimum_C(calls, meta, True)
                self.assertTrue(dg["min_record_matches_selected_call"])
                arr = np.load(d / "checkpoints" / "min_dev_arrays.npz")
                np.testing.assert_array_equal(arr["coverage_snapshot"],
                                              cov["snapshot_counts"][meta["coverage_snapshot_index"]])
            ec = json.loads((d / "economics.json").read_text())
            self.assertTrue(ec["modes"]["mean"]["pooled"]["accounting_ok"])
            for f in ("config.json", "events.jsonl", "resources.jsonl", "arrays.npz", "economics_arrays.npz"):
                self.assertTrue((d / f).exists(), f)
            with self.assertRaises(FileExistsError):
                rx.Run(mp, 99).setup()

    def test_failed_update_is_not_counted(self) -> None:
        orig = CurriculumPPO.update
        count = {"n": 0}

        def flaky(self_: CurriculumPPO, *a: Any, **k: Any) -> Dict[str, Any]:
            count["n"] += 1
            if count["n"] == 3:
                raise RuntimeError("injected update failure")
            return orig(self_, *a, **k)

        with tempfile.TemporaryDirectory() as tmp:
            mp = tmp_manifest(Path(tmp), 50, 98)
            run = rx.Run(mp, 98)
            with patch.object(CurriculumPPO, "update", flaky), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.execute(), 1)
            d = run.out
            st = json.loads((d / "status.json").read_text())
            hist, _ = common.read_jsonl(d / "history.jsonl")
            self.assertEqual((st["state"], st["last_completed_update"], len(hist)), ("failed", 2, 2))
            self.assertIn("injected update failure", (d / "traceback.txt").read_text())
            self.assertTrue((d / "coverage.npz").exists())
            self.assertIn("training", st["failure_reason"])



# ===========================================================================
# Phase-A state-sampling study (A-only, baseline vs center-weighted ES3)
# ===========================================================================

class ASamplingTests(unittest.TestCase):
    def test_center_sampler_uniform_on_B_interval(self) -> None:
        spec = spec_q(60)
        x = metrics.center_balanced(spec, 3, 200000, np.random.default_rng(0), 10.0, spec.B)
        self.assertGreaterEqual(x.min(), -220.0)
        self.assertLessEqual(x.max(), 220.0)
        counts = np.histogram(x, bins=np.linspace(-220, 220, 45))[0]
        exp = 200000 / 44
        self.assertLess(np.max(np.abs(counts - exp)), 5 * math.sqrt(exp))
        self.assertEqual(int(metrics.center_bin_mask(spec, 3, 10.0, spec.B).sum()), 44)
        self.assertEqual(int(metrics.center_bin_mask(spec_q(50), 3, 10.0, 200.0).sum()), 40)
        with self.assertRaises(ValueError):
            metrics.center_bin_mask(spec, 3, 10.0, 215.0)

    def test_mixture_draw_order_and_share(self) -> None:
        spec = spec_q(60)
        sampler = StartSampler(spec, 10.0)
        t0, d0, roles = rx.sample_starts(spec, sampler, {"3": 512}, np.random.default_rng(7), {"3": 256})
        g = np.random.default_rng(7)
        full = sampler.balanced(3, 256, g)
        cen = metrics.center_balanced(spec, 3, 256, g, 10.0, spec.B)
        perm = g.permutation(512)
        np.testing.assert_array_equal(d0, np.concatenate([full, cen])[perm])
        np.testing.assert_array_equal(roles, g.integers(0, 2, size=512))
        self.assertTrue(np.all(t0 == 3))
        share = []
        rng = np.random.default_rng(1)
        for _ in range(200):
            _, d, _ = rx.sample_starts(spec, sampler, {"3": 512}, rng, {"3": 256})
            share.append(np.mean(np.abs(d) <= spec.B))
        self.assertAlmostEqual(float(np.mean(share)), 0.75, delta=0.01)
        a = rx.sample_starts(spec, sampler, {"3": 512}, np.random.default_rng(3))
        b = rx.sample_starts(spec, sampler, {"3": 512}, np.random.default_rng(3), {})
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_outer_stage3_gain_is_wasted_effort_cost(self) -> None:
        spec = spec_q(60)
        pol = lambda t, d: 20.0 + 10.0 * np.tanh(np.asarray(d, dtype=float) / 150.0)
        res = dpv.verify(pol, w_h=6, w_l=2, k=spec.k, q=60, T=3, cfg=dpv.FINAL_CONFIG)
        s = res.stages[3]
        outer = np.abs(s.d_grid) > spec.B
        np.testing.assert_allclose((s.v_br - s.v_mean)[outer], spec.k * s.e_hat[outer] ** 2, atol=1e-12)
        np.testing.assert_array_equal(s.a_br[outer], 0.0)
        m = metrics.stage_region_metrics(s.d_grid, s.v_br - s.v_mean, None, spec.B, spec.dw)
        self.assertEqual((m["n_center"], m["n_outer"]), (221, 220))
        self.assertAlmostEqual(m["outer_max_over_dw"], float(np.max(spec.k * s.e_hat[outer] ** 2)) / 4.0, places=14)

    def test_region_metrics_synthetic(self) -> None:
        d = np.array([-440.0, -221.0, -220.0, 0.0, 220.0, 221.0, 440.0])
        g = np.array([0.4, 0.8, 0.2, 0.1, 0.3, 0.6, 0.4])
        sn = np.array([0.01, 0.02, 0.03, 0.05, 0.02, 0.01, 0.0])
        m = metrics.stage_region_metrics(d, g, sn, 220.0, 4.0)
        self.assertEqual((m["n_center"], m["n_outer"]), (3, 4))
        self.assertEqual((m["all_max_over_dw"], m["all_argmax_d"]), (0.2, -221.0))
        self.assertEqual((m["center_max_over_dw"], m["center_argmax_d"]), (0.075, 220.0))
        self.assertAlmostEqual(m["outer_mean_over_dw"], (0.4 + 0.8 + 0.6 + 0.4) / 4 / 4)
        self.assertEqual((m["center_max_std_norm"], m["center_max_std_norm_d"]), (0.05, 0.0))

    def test_arm_manifests(self) -> None:
        pilot = bm.build_manifest("pilot", 60, [10401])["resolved"]
        arms = {a: bm.build_a_sampling_manifest("asamp", 60, bm.SEEDS["asamp"][60], a) for a in bm.A_SAMPLING_ARMS}
        for arm, m in arms.items():
            cfg = m["resolved"]
            self.assertEqual([p["name"] for p in cfg["phases"]], ["A"])
            A = cfg["phases"][0]
            pA = pilot["phases"][0]
            for k in ("cap", "check_every", "start_counts", "active_stages", "concentration_stages", "criterion"):
                self.assertEqual(A[k], pA[k])
            for k in ("ppo", "verifier", "final_certification", "concentration_threshold", "snapshot_every", "rng",
                      "game", "derived"):
                self.assertEqual(cfg[k], pilot[k])
            self.assertFalse(cfg["economics"]["enabled"])
            self.assertEqual(cfg["study"]["arm"], arm)
            self.assertIn("all 3 seed pairs", cfg["study"]["decision_rule"])
            self.assertEqual(A["per_update"]["joint_environment_steps"], 512)
        self.assertNotIn("center_es", arms["baseline"]["resolved"]["phases"][0])
        self.assertEqual(arms["center"]["resolved"]["phases"][0]["center_es"], {"3": 256})
        ex = arms["center"]["resolved"]["phases"][0]["expected_direct_es_per_bin_per_update"]["3"]
        self.assertAlmostEqual(ex["expected_center_share"], 0.75)
        self.assertAlmostEqual(ex["outer_bin"] * 400, 1163.636, places=3)
        ids = [r["run_id"] for m in arms.values() for r in m["runs"]]
        self.assertEqual(len(set(ids)), 6)
        with self.assertRaises(ValueError):
            bm.build_a_sampling_manifest("asamp", 60, [10401], "center")
        with self.assertRaises(ValueError):
            bm.build_manifest("formal", 60, [10431], {"resolved_non_q": {}, "seeds_per_q": 10})

    def test_baseline_arm_reproduces_pilot_phase_A(self) -> None:
        pilot_hist = E_DIR / "runs" / "t3_pilot_q60_s10401" / "history.jsonl"
        if not pilot_hist.exists():
            self.skipTest("pilot run data not present")
        ref, _ = common.read_jsonl(pilot_hist)
        m = bm.build_a_sampling_manifest("asamp", 60, bm.SEEDS["asamp"][60], "baseline")
        with tempfile.TemporaryDirectory() as tmp:
            m["resolved"]["phases"][0]["cap"] = 3
            m["runs"] = [dict(m["runs"][0], seed=10401, run_id="unit_asamp_repro", output_dir=str(Path(tmp) / "r"))]
            mp = Path(tmp) / "m.json"
            mp.write_text(json.dumps(m))
            run = rx.Run(mp, 10401)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.execute(), 0)
            got, _ = common.read_jsonl(run.out / "history.jsonl")
            for a, b in zip(got, ref[:3]):
                for k in ("policy_loss", "value_loss", "kl_final_epoch", "mean_episode_return", "clip_frac"):
                    self.assertEqual(a[k], b[k], k)
            ec = json.loads((run.out / "economics.json").read_text())
            self.assertTrue(ec["skipped"])

    def test_tiny_center_arm_run(self) -> None:
        m = bm.build_a_sampling_manifest("asamp_smoke", 60, bm.SEEDS["asamp_smoke"][60], "center")
        with tempfile.TemporaryDirectory() as tmp:
            m["runs"] = [dict(m["runs"][0], run_id="unit_asamp_center", output_dir=str(Path(tmp) / "r"))]
            mp = Path(tmp) / "m.json"
            mp.write_text(json.dumps(m))
            run = rx.Run(mp, 10430)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.execute(), 0)
            d = run.out
            fe = json.loads((d / "final_eval.json").read_text())
            for tier in ("development", "final"):
                r = fe["tiers"][tier]["stage3_regions"]
                self.assertEqual(r["all_max_over_dw"], max(r["center_max_over_dw"], r["outer_max_over_dw"]))
            self.assertIn("outer_max_std_norm", fe["dense_stage3_conc_regions"])
            calls, _ = common.read_jsonl(d / "verifier_calls.jsonl")
            self.assertEqual([c["local_update"] for c in calls], [1, 2])
            self.assertIn("stage3_regions", calls[-1])
            st = json.loads((d / "status.json").read_text())
            self.assertEqual((st["state"], st["search_outcome"]), ("done", "C_not_reached"))
            self.assertTrue(json.loads((d / "economics.json").read_text())["skipped"])
            cov = np.load(d / "coverage.npz")
            self.assertEqual(int(cov["phase_totals"][0].sum()), 3 * 64)   # visits + raw + learner, 2 updates x 32


# ===========================================================================
# Paired full A/B/C validation (phase-A sampling differs only)
# ===========================================================================

class ABCValidationTests(unittest.TestCase):
    def test_arm_manifests_differ_only_in_phase_A_sampling(self) -> None:
        pilot = bm.build_manifest("pilot", 60, [10401])["resolved"]
        arms = {a: bm.build_abc_arm_manifest("asamp_abc", 60, bm.SEEDS["asamp_abc"][60], a)["resolved"]
                for a in bm.A_SAMPLING_ARMS}
        for arm, cfg in arms.items():
            self.assertEqual([p["name"] for p in cfg["phases"]], ["A", "B", "C"])
            self.assertEqual(cfg["phases"][1:], pilot["phases"][1:])          # B and C untouched
            a0 = {k: v for k, v in cfg["phases"][0].items() if k not in ("center_es", "expected_direct_es_per_bin_per_update")}
            p0 = {k: v for k, v in pilot["phases"][0].items() if k != "expected_direct_es_per_bin_per_update"}
            self.assertEqual(a0, p0)
            rest = {k: v for k, v in cfg.items() if k not in ("phases", "study")}
            self.assertEqual(rest, {k: v for k, v in pilot.items() if k != "phases"})
            self.assertTrue(cfg["economics"].get("enabled", True))
            self.assertEqual(cfg["economics"]["episodes_per_replicate"], 200000)
            self.assertEqual(cfg["study"]["phases_run"], ["A", "B", "C"])
            self.assertIn("certified", cfg["study"]["primary_endpoint"])
        self.assertEqual(arms["center"]["phases"][0]["center_es"], {"3": 256})
        self.assertNotIn("center_es", arms["baseline"]["phases"][0])
        for ph in arms["center"]["phases"][1:]:
            self.assertNotIn("center_es", ph)
        with self.assertRaises(ValueError):
            bm.build_abc_arm_manifest("asamp_abc", 60, [10431], "center")      # A-only study seed
        sm = bm.build_abc_arm_manifest("asamp_abc_smoke", 60, [10440], "center")["resolved"]
        self.assertEqual(sm["phases"][0]["center_es"], {"3": 16})
        self.assertEqual([p["cap"] for p in sm["phases"]], [2, 2, 2])

    def test_rates_grouped_per_arm(self) -> None:
        rows = [{"q": 60, "arm": "baseline", "state": "done", "has_candidate": False, "certification": None},
                {"q": 60, "arm": "center", "state": "done", "has_candidate": True, "certification": "certified"},
                {"q": 60, "arm": "center", "state": "done", "has_candidate": True, "certification": "not_certified"}]
        g = make_report.rate_groups(rows, "debug")
        self.assertEqual(set(g), {"60_baseline", "60_center"})
        self.assertEqual(g["60_center"]["primary"]["candidate_discovery"]["k"], 2)
        self.assertEqual(g["60_center"]["primary"]["conditional_certification"]["n"], 2)
        self.assertEqual(g["60_baseline"]["primary"]["candidate_discovery"]["k"], 0)
        self.assertEqual(set(make_report.rate_groups([dict(r, arm=None) for r in rows], "x")), {60})

    def test_tiny_center_arm_full_run(self) -> None:
        m = bm.build_abc_arm_manifest("asamp_abc_smoke", 60, bm.SEEDS["asamp_abc_smoke"][60], "center")
        m["resolved"]["economics"].update(episodes_per_replicate=100, chunk_size=64)
        with tempfile.TemporaryDirectory() as tmp:
            m["runs"] = [dict(m["runs"][0], run_id="unit_abc_center", output_dir=str(Path(tmp) / "r"))]
            mp = Path(tmp) / "m.json"
            mp.write_text(json.dumps(m))
            run = rx.Run(mp, 10440)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.execute(), 0)
            d = run.out
            hist, _ = common.read_jsonl(d / "history.jsonl")
            self.assertEqual([h["n_environment_steps"] for h in hist][:4], [32, 32, 64, 64])
            ec = json.loads((d / "economics.json").read_text())
            self.assertIn("modes", ec)
            fe = json.loads((d / "final_eval.json").read_text())
            self.assertIn("stage3_regions", fe["tiers"]["final"])
            calls, _ = common.read_jsonl(d / "verifier_calls.jsonl")
            self.assertTrue(all("stage3_regions" in c for c in calls))

if __name__ == "__main__":
    unittest.main(verbosity=2)
