#!/usr/bin/env python3
"""Diagnostic script for three_players, different_cost, different_ability experiments.

Checks:
1. Participation constraint (per experiment, per q, per player)
2. Convergence status (effort gap, exploitability, stop reason)
3. Data integrity (truncated runs, missing fields)
4. Theory consistency (theory.py vs JSON vs independent formula)

Usage:
    python tools/diagnose_all.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.prob import p_from_diff, win_prob_three_players
from utils.theory import (
    e_star_two_players,
    e_star_three_players,
    e_star_two_players_asymmetric_cost,
    e_star_two_players_different_ability,
    p_win_different_ability,
)

# ============================================================================
# Game parameters
# ============================================================================
W_H = 6.5
W_L = 3.0
K = 0.0004
K1 = 0.0004
K2 = 0.00055
L1 = 10.0
L2 = 5.0
Q_VALUES = [25.0, 40.0, 55.0]


# ============================================================================
# Independent P(win) implementations (cross-validation against prob.py)
# ============================================================================

def _independent_p_win_2p(d: float, q: float) -> float:
    """P(player wins) given effort difference d = e_i - e_j, noise U[-q, q].

    Derived from triangular distribution of eps1 - eps2 on [-2q, 2q].
    """
    if d <= -2.0 * q:
        return 0.0
    if d >= 2.0 * q:
        return 1.0
    # Piecewise: 0.5 + d/(2q) - d|d|/(8q^2)
    return 0.5 + d / (2.0 * q) - d * abs(d) / (8.0 * q * q)


def _uniform_cdf(x: float, q: float) -> float:
    """CDF of a single U[-q, q] variable: P(eps < x)."""
    if x <= -q:
        return 0.0
    if x >= q:
        return 1.0
    return (x + q) / (2.0 * q)


def _independent_p_win_3p(e_i: float, e_j: float, e_k: float, q: float) -> float:
    """P(player i wins) in 3-player tournament with U[-q, q] noise.

    Player i wins iff: e_i + eps_i > e_j + eps_j AND e_i + eps_i > e_k + eps_k.
    Numerical integration over eps_i ~ U[-q, q].

    Conditioned on eps_i, P(beat j | eps_i) = P(eps_j < e_i + eps_i - e_j)
    which is the CDF of U[-q, q] (single uniform), NOT the triangular distribution.
    """
    n_steps = 10000
    step = 2.0 * q / n_steps
    total = 0.0
    for i in range(n_steps + 1):
        eps_i = -q + (2.0 * q * i / n_steps)
        output_i = e_i + eps_i
        # P(eps_j < output_i - e_j) — single uniform CDF
        p_beat_j = _uniform_cdf(output_i - e_j, q)
        # P(eps_k < output_i - e_k) — single uniform CDF
        p_beat_k = _uniform_cdf(output_i - e_k, q)
        total += p_beat_j * p_beat_k
    # Trapezoidal integration over uniform density 1/(2q)
    return total * step / (2.0 * q)


def _independent_p_win_diff_ability(
    e1: float, e2: float, l1: float, l2: float, q: float,
) -> float:
    """P(player 1 wins) in different-ability model.

    y_i = e_i + l_i + eps_i, eps ~ U[-q, q].
    P(y1 > y2) = P(eps1 - eps2 > (e2 + l2) - (e1 + l1)).
    """
    d = (e1 + l1) - (e2 + l2)
    return _independent_p_win_2p(d, q)


# ============================================================================
# Data loading
# ============================================================================

@dataclass
class RunData:
    """Unified run data across all experiment types."""
    experiment: str
    filename: str
    q: float
    seed: int
    ablation: str
    # Efforts
    final_effort1: float = float("nan")
    final_effort2: float = float("nan")
    final_effort3: float = float("nan")  # three_players only
    theoretical_effort1: float = float("nan")
    theoretical_effort2: float = float("nan")
    # Gaps
    gap1: float = float("nan")
    gap2: float = float("nan")
    # Exploitability
    final_exploit_max: float = float("nan")
    final_exploit_1: float = float("nan")
    final_exploit_2: float = float("nan")
    br_effort_1: float = float("nan")
    br_effort_2: float = float("nan")
    # Convergence
    stop_reason: str = "unknown"
    num_updates: int = 0
    total_steps: int = 0
    exploit_streak: int = 0
    # Raw params from JSON
    params: Dict[str, Any] = field(default_factory=dict)


def load_results(experiment: str) -> List[RunData]:
    """Load all baseline convergence JSONs for an experiment type."""
    conv_dir = os.path.join(PROJECT_ROOT, "results", experiment, "convergence")
    prefix_map = {
        "three_players": "ppo_3p_",
        "different_cost": "different_cost_ppo_",
        "different_ability": "different_ability_ppo_",
    }
    prefix = prefix_map[experiment]
    pattern = os.path.join(conv_dir, f"{prefix}q*_seed*_baseline_convergence.json")
    files = sorted(glob.glob(pattern))

    runs = []
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)

        r = RunData(
            experiment=experiment,
            filename=os.path.basename(fpath),
            q=d.get("q", 0.0),
            seed=d.get("seed", 0),
            ablation=d.get("ablation_name", "baseline"),
        )

        if experiment == "three_players":
            fr = d.get("final_results", {})
            r.final_effort1 = fr.get("final_effort", float("nan"))
            r.theoretical_effort1 = d.get("theoretical_effort", float("nan"))
            r.gap1 = fr.get("abs_error", float("nan"))
            steps = d.get("steps", [])
            r.num_updates = len(steps)
            r.total_steps = steps[-1] if steps else 0
            r.stop_reason = d.get("stop_reason", "unknown")

        elif experiment == "different_cost":
            final = d.get("final", {})
            theo = d.get("theoretical", {})
            r.final_effort1 = final.get("effort1", float("nan"))
            r.final_effort2 = final.get("effort2", float("nan"))
            r.theoretical_effort1 = theo.get("effort1", float("nan"))
            r.theoretical_effort2 = theo.get("effort2", float("nan"))
            r.gap1 = final.get("gap1", float("nan"))
            r.gap2 = final.get("gap2", float("nan"))
            r.final_exploit_max = d.get("final_exploit_max", float("nan"))
            r.final_exploit_1 = d.get("final_exploit_1", float("nan"))
            r.final_exploit_2 = d.get("final_exploit_2", float("nan"))
            r.br_effort_1 = d.get("final_br_effort_1", float("nan"))
            r.br_effort_2 = d.get("final_br_effort_2", float("nan"))
            r.stop_reason = d.get("stop_reason", "unknown")
            r.exploit_streak = d.get("joint_exploit_ok_streak", 0)
            steps = d.get("history", {}).get("steps", [])
            r.num_updates = len(steps)
            r.total_steps = steps[-1] if steps else 0
            r.params = {"k1": d.get("k1", K1), "k2": d.get("k2", K2),
                        "w_h": d.get("w_h", W_H), "w_l": d.get("w_l", W_L)}

        elif experiment == "different_ability":
            final = d.get("final", {})
            theo = d.get("theoretical", {})
            r.final_effort1 = final.get("effort", float("nan"))
            r.final_effort2 = final.get("effort", float("nan"))  # symmetric effort
            r.theoretical_effort1 = theo.get("effort", float("nan"))
            r.theoretical_effort2 = theo.get("effort", float("nan"))
            r.gap1 = final.get("gap", float("nan"))
            r.gap2 = final.get("gap", float("nan"))
            r.final_exploit_max = d.get("final_exploit_max", float("nan"))
            r.final_exploit_1 = d.get("final_exploit_1", float("nan"))
            r.final_exploit_2 = d.get("final_exploit_2", float("nan"))
            r.br_effort_1 = d.get("final_br_effort_1", float("nan"))
            r.br_effort_2 = d.get("final_br_effort_2", float("nan"))
            r.stop_reason = d.get("stop_reason", "unknown")
            r.exploit_streak = d.get("joint_exploit_ok_streak", 0)
            steps = d.get("history", {}).get("steps", [])
            r.num_updates = len(steps)
            r.total_steps = steps[-1] if steps else 0
            r.params = {"k": d.get("k", K), "l1": d.get("l1", L1), "l2": d.get("l2", L2),
                        "w_h": d.get("w_h", W_H), "w_l": d.get("w_l", W_L)}

        runs.append(r)
    return runs


# ============================================================================
# Check 1: Participation constraint
# ============================================================================

@dataclass
class ParticipationResult:
    experiment: str
    q: float
    player: str
    e_star: float
    eu_estar: float
    eu_zero: float
    p_win_zero: float
    valid: bool
    deviation_gain: float


def check_participation_three_players(q: float) -> List[ParticipationResult]:
    """Check participation constraint for symmetric 3-player tournament."""
    e = e_star_three_players(q, W_H, W_L, K)
    cost = K * e * e
    # Symmetric: P(win) = 1/3
    eu_estar = W_L + (1.0 / 3.0) * (W_H - W_L) - cost

    # Deviation to e=0: use independent implementation
    p_win_zero_indep = _independent_p_win_3p(0.0, e, e, q)
    eu_zero = W_L + p_win_zero_indep * (W_H - W_L)

    valid = eu_estar >= eu_zero
    return [ParticipationResult(
        experiment="three_players", q=q, player="symmetric",
        e_star=e, eu_estar=eu_estar, eu_zero=eu_zero,
        p_win_zero=p_win_zero_indep, valid=valid,
        deviation_gain=eu_zero - eu_estar,
    )]


def check_participation_different_cost(q: float) -> List[ParticipationResult]:
    """Check participation constraint for asymmetric-cost 2-player tournament."""
    e1, e2 = e_star_two_players_asymmetric_cost(q, W_H, W_L, K1, K2)
    results = []
    for player, e_self, e_opp, k in [("P1 (k=0.0004)", e1, e2, K1),
                                       ("P2 (k=0.00055)", e2, e1, K2)]:
        cost = k * e_self * e_self
        p_win_eq = _independent_p_win_2p(e_self - e_opp, q)
        eu_estar = W_L + p_win_eq * (W_H - W_L) - cost

        # Deviation to e=0
        p_win_zero = _independent_p_win_2p(0.0 - e_opp, q)
        eu_zero = W_L + p_win_zero * (W_H - W_L)

        valid = eu_estar >= eu_zero
        results.append(ParticipationResult(
            experiment="different_cost", q=q, player=player,
            e_star=e_self, eu_estar=eu_estar, eu_zero=eu_zero,
            p_win_zero=p_win_zero, valid=valid,
            deviation_gain=eu_zero - eu_estar,
        ))
    return results


def check_participation_different_ability(q: float) -> List[ParticipationResult]:
    """Check participation constraint for different-ability 2-player tournament."""
    e = e_star_two_players_different_ability(q, W_H, W_L, K, L1, L2)
    cost = K * e * e
    results = []
    for player, l_self, l_opp in [("P1 (l=10, stronger)", L1, L2),
                                    ("P2 (l=5, weaker)", L2, L1)]:
        # At equilibrium (both play e), P(win) depends on ability difference
        p_win_eq = _independent_p_win_diff_ability(e, e, l_self, l_opp, q)
        eu_estar = W_L + p_win_eq * (W_H - W_L) - cost

        # Deviation to e=0
        p_win_zero = _independent_p_win_diff_ability(0.0, e, l_self, l_opp, q)
        eu_zero = W_L + p_win_zero * (W_H - W_L)

        valid = eu_estar >= eu_zero
        results.append(ParticipationResult(
            experiment="different_ability", q=q, player=player,
            e_star=e, eu_estar=eu_estar, eu_zero=eu_zero,
            p_win_zero=p_win_zero, valid=valid,
            deviation_gain=eu_zero - eu_estar,
        ))
    return results


# ============================================================================
# Check 2: Convergence status
# ============================================================================

def check_convergence(runs: List[RunData]) -> List[RunData]:
    """Just returns runs sorted for reporting. Analysis done in formatting."""
    return sorted(runs, key=lambda r: (r.q, r.seed))


# ============================================================================
# Check 3: Data integrity
# ============================================================================

@dataclass
class IntegrityIssue:
    filename: str
    issue: str


def check_integrity(runs: List[RunData], expected_updates: int = 1500) -> List[IntegrityIssue]:
    """Check for truncated runs, missing fields, NaN values."""
    issues = []
    for r in runs:
        if r.num_updates < expected_updates and r.stop_reason not in ("exploitability",):
            issues.append(IntegrityIssue(
                r.filename,
                f"Truncated: {r.num_updates}/{expected_updates} updates, stop_reason={r.stop_reason}",
            ))
        if r.stop_reason == "unknown":
            issues.append(IntegrityIssue(r.filename, "Missing stop_reason field"))
        if math.isnan(r.final_effort1):
            issues.append(IntegrityIssue(r.filename, "final_effort1 is NaN"))
        if math.isnan(r.theoretical_effort1):
            issues.append(IntegrityIssue(r.filename, "theoretical_effort1 is NaN"))
    return issues


# ============================================================================
# Check 4: Theory consistency
# ============================================================================

@dataclass
class TheoryCheck:
    experiment: str
    q: float
    player: str
    json_value: float
    theory_py_value: float
    formula_value: float
    consistent: bool  # all within tolerance


def check_theory_three_players(q: float, runs: List[RunData]) -> List[TheoryCheck]:
    """Cross-check three_players theoretical effort: JSON vs theory.py vs formula."""
    # theory.py
    e_theory_py = e_star_three_players(q, W_H, W_L, K)
    # Independent formula: e* = (w_H - w_L) / (4kq)
    e_formula = (W_H - W_L) / (4.0 * K * q)
    # JSON value (from any run with this q)
    json_vals = [r.theoretical_effort1 for r in runs
                 if r.q == q and not math.isnan(r.theoretical_effort1)]
    e_json = json_vals[0] if json_vals else float("nan")

    tol = 0.01
    consistent = (abs(e_theory_py - e_formula) < tol and
                  (math.isnan(e_json) or abs(e_json - e_formula) < tol))
    return [TheoryCheck("three_players", q, "symmetric",
                        e_json, e_theory_py, e_formula, consistent)]


def check_theory_different_cost(q: float, runs: List[RunData]) -> List[TheoryCheck]:
    """Cross-check different_cost theoretical efforts."""
    e1_tp, e2_tp = e_star_two_players_asymmetric_cost(q, W_H, W_L, K1, K2)
    # Independent formula
    w_gap = W_H - W_L
    denom = 8.0 * K1 * K2 * q * q - (K1 - K2) * w_gap
    e1_f = (2.0 * K2 * q * w_gap) / denom
    e2_f = (2.0 * K1 * q * w_gap) / denom
    # JSON
    q_runs = [r for r in runs if r.q == q]
    e1_j = q_runs[0].theoretical_effort1 if q_runs else float("nan")
    e2_j = q_runs[0].theoretical_effort2 if q_runs else float("nan")

    tol = 0.01
    results = []
    for player, j, tp, f in [("P1 (k1)", e1_j, e1_tp, e1_f),
                               ("P2 (k2)", e2_j, e2_tp, e2_f)]:
        consistent = (abs(tp - f) < tol and
                      (math.isnan(j) or abs(j - f) < tol))
        results.append(TheoryCheck("different_cost", q, player, j, tp, f, consistent))
    return results


def check_theory_different_ability(q: float, runs: List[RunData]) -> List[TheoryCheck]:
    """Cross-check different_ability theoretical effort."""
    e_tp = e_star_two_players_different_ability(q, W_H, W_L, K, L1, L2)
    # Independent formula: e* = ((2q - (l1-l2)) * (w_H - w_L)) / (8kq^2)
    w_gap = W_H - W_L
    delta_l = L1 - L2
    e_f = max(0.0, ((2.0 * q - delta_l) * w_gap) / (8.0 * K * q * q))
    # JSON
    q_runs = [r for r in runs if r.q == q]
    e_j = q_runs[0].theoretical_effort1 if q_runs else float("nan")

    tol = 0.01
    consistent = (abs(e_tp - e_f) < tol and
                  (math.isnan(e_j) or abs(e_j - e_f) < tol))
    return [TheoryCheck("different_ability", q, "symmetric",
                        e_j, e_tp, e_f, consistent)]


# ============================================================================
# Cross-validation: prob.py vs independent P(win)
# ============================================================================

@dataclass
class ProbCheck:
    experiment: str
    description: str
    prob_py_value: float
    independent_value: float
    match: bool


def cross_validate_prob() -> List[ProbCheck]:
    """Cross-check prob.py functions against independent implementations."""
    checks = []
    tol = 0.001

    # 2-player: p_from_diff
    for d, q in [(0.0, 25.0), (10.0, 40.0), (-30.0, 55.0), (87.5, 25.0)]:
        pp = float(p_from_diff(d, q))
        indep = _independent_p_win_2p(d, q)
        checks.append(ProbCheck(
            "two_players", f"p_from_diff(d={d}, q={q})",
            pp, indep, abs(pp - indep) < tol,
        ))

    # 3-player: win_prob_three_players
    for ei, ej, ek, q in [(50, 50, 50, 40), (0, 87.5, 87.5, 25), (54.7, 54.7, 54.7, 40)]:
        pp = win_prob_three_players(ei, ej, ek, q)
        indep = _independent_p_win_3p(ei, ej, ek, q)
        checks.append(ProbCheck(
            "three_players", f"win_prob_3p({ei},{ej},{ek},q={q})",
            pp, indep, abs(pp - indep) < tol,
        ))

    # different_ability: p_win_different_ability
    for e1, e2, l1, l2, q in [(50, 50, 10, 5, 40), (0, 50, 10, 5, 40), (78.75, 78.75, 10, 5, 25)]:
        pp = p_win_different_ability(e1, e2, l1, l2, q)
        indep = _independent_p_win_diff_ability(e1, e2, l1, l2, q)
        checks.append(ProbCheck(
            "different_ability", f"p_win_da(e1={e1},e2={e2},l={l1}/{l2},q={q})",
            pp, indep, abs(pp - indep) < tol,
        ))

    return checks


# ============================================================================
# Report formatting
# ============================================================================

def format_report(
    participation: Dict[str, List[ParticipationResult]],
    runs: Dict[str, List[RunData]],
    integrity: Dict[str, List[IntegrityIssue]],
    theory: Dict[str, List[TheoryCheck]],
    prob_checks: List[ProbCheck],
) -> str:
    lines = []
    lines.append("# Diagnostic Report: three_players, different_cost, different_ability")
    lines.append("")
    lines.append("Generated from `tools/diagnose_all.py`")
    lines.append("")

    # 0. Probability cross-validation
    lines.append("## 0. Probability Cross-Validation (prob.py vs independent)")
    lines.append("")
    lines.append("| Experiment | Test | prob.py | Independent | Match? |")
    lines.append("|---|---|---|---|---|")
    for pc in prob_checks:
        m = "YES" if pc.match else "**NO**"
        lines.append(f"| {pc.experiment} | {pc.description} | {pc.prob_py_value:.6f} | {pc.independent_value:.6f} | {m} |")
    lines.append("")

    # Per experiment
    for exp in ["three_players", "different_cost", "different_ability"]:
        lines.append(f"---")
        lines.append(f"## {exp}")
        lines.append("")

        # 1. Participation constraint
        lines.append(f"### 1. Participation Constraint")
        lines.append("")
        lines.append("| q | Player | e* | EU(e*) | EU(0) | P(win\\|0) | Dev gain | Valid? |")
        lines.append("|---|--------|-----|--------|-------|-----------|---------|--------|")
        for p in participation.get(exp, []):
            v = "YES" if p.valid else "**NO**"
            lines.append(
                f"| {p.q:.0f} | {p.player} | {p.e_star:.2f} | "
                f"{p.eu_estar:.4f} | {p.eu_zero:.4f} | {p.p_win_zero:.6f} | "
                f"{p.deviation_gain:.4f} | {v} |"
            )
        lines.append("")

        # 2. Convergence status
        lines.append(f"### 2. Convergence Status")
        lines.append("")
        exp_runs = runs.get(exp, [])
        if exp == "three_players":
            lines.append("| q | seed | stop | effort | e* | gap | updates |")
            lines.append("|---|------|------|--------|-----|-----|---------|")
            for r in sorted(exp_runs, key=lambda x: (x.q, x.seed)):
                lines.append(
                    f"| {r.q:.0f} | {r.seed} | {r.stop_reason} | "
                    f"{r.final_effort1:.2f} | {r.theoretical_effort1:.2f} | "
                    f"{r.gap1:.2f} | {r.num_updates} |"
                )
        elif exp == "different_cost":
            lines.append("| q | seed | stop | e1 | e1* | gap1 | e2 | e2* | gap2 | exploit | streak |")
            lines.append("|---|------|------|-----|------|------|-----|------|------|---------|--------|")
            for r in sorted(exp_runs, key=lambda x: (x.q, x.seed)):
                ex = f"{r.final_exploit_max:.4f}" if not math.isnan(r.final_exploit_max) else "N/A"
                lines.append(
                    f"| {r.q:.0f} | {r.seed} | {r.stop_reason} | "
                    f"{r.final_effort1:.2f} | {r.theoretical_effort1:.2f} | {r.gap1:.2f} | "
                    f"{r.final_effort2:.2f} | {r.theoretical_effort2:.2f} | {r.gap2:.2f} | "
                    f"{ex} | {r.exploit_streak} |"
                )
        elif exp == "different_ability":
            lines.append("| q | seed | stop | effort | e* | gap | exploit | streak |")
            lines.append("|---|------|------|--------|-----|-----|---------|--------|")
            for r in sorted(exp_runs, key=lambda x: (x.q, x.seed)):
                ex = f"{r.final_exploit_max:.4f}" if not math.isnan(r.final_exploit_max) else "N/A"
                lines.append(
                    f"| {r.q:.0f} | {r.seed} | {r.stop_reason} | "
                    f"{r.final_effort1:.2f} | {r.theoretical_effort1:.2f} | {r.gap1:.2f} | "
                    f"{ex} | {r.exploit_streak} |"
                )
        lines.append("")

        # Summary
        lines.append("#### Summary per q")
        lines.append("")
        lines.append("| q | seeds | converged | mean_gap |")
        lines.append("|---|-------|-----------|----------|")
        for q in Q_VALUES:
            q_runs = [r for r in exp_runs if r.q == q]
            n = len(q_runs)
            converged = sum(1 for r in q_runs if r.stop_reason == "exploitability")
            gaps = [r.gap1 for r in q_runs if not math.isnan(r.gap1)]
            mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
            mg = f"{mean_gap:.2f}" if not math.isnan(mean_gap) else "N/A"
            lines.append(f"| {q:.0f} | {n} | {converged}/{n} | {mg} |")
        lines.append("")

        # 3. Data integrity
        issues = integrity.get(exp, [])
        lines.append(f"### 3. Data Integrity")
        lines.append("")
        if issues:
            for iss in issues:
                lines.append(f"- **{iss.filename}**: {iss.issue}")
        else:
            lines.append("No issues found.")
        lines.append("")

        # 4. Theory consistency
        lines.append(f"### 4. Theory Consistency")
        lines.append("")
        lines.append("| q | Player | JSON | theory.py | Formula | Consistent? |")
        lines.append("|---|--------|------|-----------|---------|-------------|")
        for tc in theory.get(exp, []):
            j = f"{tc.json_value:.4f}" if not math.isnan(tc.json_value) else "N/A"
            c = "YES" if tc.consistent else "**NO**"
            lines.append(
                f"| {tc.q:.0f} | {tc.player} | {j} | "
                f"{tc.theory_py_value:.4f} | {tc.formula_value:.4f} | {c} |"
            )
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # Cross-validate probability functions first
    prob_checks = cross_validate_prob()

    participation: Dict[str, List[ParticipationResult]] = {}
    all_runs: Dict[str, List[RunData]] = {}
    integrity: Dict[str, List[IntegrityIssue]] = {}
    theory: Dict[str, List[TheoryCheck]] = {}

    for exp in ["three_players", "different_cost", "different_ability"]:
        print(f"Loading {exp}...")
        runs = load_results(exp)
        print(f"  Found {len(runs)} baseline runs")
        all_runs[exp] = check_convergence(runs)
        integrity[exp] = check_integrity(runs)

        # Participation constraint
        part = []
        for q in Q_VALUES:
            if exp == "three_players":
                part.extend(check_participation_three_players(q))
            elif exp == "different_cost":
                part.extend(check_participation_different_cost(q))
            elif exp == "different_ability":
                part.extend(check_participation_different_ability(q))
        participation[exp] = part

        # Theory consistency
        th = []
        for q in Q_VALUES:
            if exp == "three_players":
                th.extend(check_theory_three_players(q, runs))
            elif exp == "different_cost":
                th.extend(check_theory_different_cost(q, runs))
            elif exp == "different_ability":
                th.extend(check_theory_different_ability(q, runs))
        theory[exp] = th

    report = format_report(participation, all_runs, integrity, theory, prob_checks)
    print(report)

    report_path = os.path.join(
        PROJECT_ROOT, "docs", "tasks", "diagnose-all-experiments",
        "diagnostic_report.md",
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
