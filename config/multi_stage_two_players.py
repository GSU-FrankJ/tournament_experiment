# Canonical configuration for the multi-stage tournament experiments
# (docs/Experiments Plan_Multi-stage.md + docs/tasks/multistage-tel-ppo/).
#
# This file SUPERSEDES config/two_stage_two_players.py, which describes a
# different game (per-stage prize flow, logit win model, stage weights) and
# must not be used for the multi-stage plan. It also resolves the parameter
# conflict between the plan document and the old config:
#
#   source                        w_h  w_l  k        q         cost conv.
#   plan doc (benchmark section)  --   --   1/3500   50        (k/2) e^2, DW=2
#   plan doc (parameter table)    6    2    1/3500   50        k e^2, DW=4
#   old two_stage config          6.5  3.0  0.0004   25/40/55  k e^2 (wrong game)
#
# Canonical choice: REPO convention c(e) = k e^2 with w_h=6, w_l=2 (DW=4).
# Under this convention the plan's formulas convert via k -> 2k; the effort
# targets are unchanged (g1 = 46.67, g2(0) = 70 at q=50).
#
# VALIDITY: the closed-form benchmark requires q > q_crit = 41.83 for these
# prizes/costs (binding constraint: SOC, q_soc = sqrt(DW/8k)). q=35 and q=40
# are INVALID (numerically confirmed: the symmetric candidate becomes a local
# minimum and give-up deviations are profitable). Hence q_list uses 45/50/55.
# Run `python -m config.multi_stage_two_players` to print the validation
# report; runners must call validate() before training.

from typing import Any, Dict

from utils.theory_multistage import (
    g1_two_stage,
    g2_two_stage,
    q_crit,
    validate_two_stage_params,
)

config: Dict[str, Any] = {
    # --- Game (terminal-reward dynamic Lazear-Rosen, plan section 1) ---
    "num_players": 2,
    "T": 2,                      # current phase horizon; extensions: 3, 4, 5
    "T_list": [2, 3, 4, 5],
    "w_h": 6.0,
    "w_l": 2.0,
    "k": 1.0 / 3500.0,           # cost convention: c(e) = k * e^2 (repo invariant)
    "q": 50.0,
    "q_list": [45.0, 50.0, 55.0],  # all > q_crit = 41.83; q<=40 fails SOC
    "effort_range": [0.0, 100.0],
    "u_bar": 0.0,                # outside option for the participation screen
    # Terminal reward: winner w_h, loser w_l, tie (w_h+w_l)/2 (measure zero)

    # --- Training rewards (sampled outcomes ONLY, repo invariant) ---
    # r_t = -k e_t^2 for t < T; r_T = R(realized final gap) - k e_T^2.
    # Closed-form win probabilities must never enter the env step reward.

    # --- RL return specification (plan section 3.4) ---
    "gamma": 1.0,                # finite-horizon economic payoff; NOT the agent
    "gae_lambda": 1.0,           # default 0.99/0.95 must be overridden
    "gae_lambda_robustness": 0.95,

    # --- State input (plan section 3.1) ---
    # s_t = [t/T, d_t / (q * sqrt(t))]; per-cell training, so constant
    # parameter features (w_h, w_l, k, q) are omitted.

    # --- Exploring starts (owner decision 2026-07-09: keep full MPE claim) ---
    # Episodes reset from random (t, d) so off-path states receive gradient
    # signal; MPE requires best response at ALL states, and the verifier
    # scans the full grid. rho = fraction of on-path (t=1, d=0) episodes.
    "exploring_starts": True,
    "es_on_path_fraction": 0.5,
    "es_d_range_factor": 1.0,    # start-gap range: +/- factor * 2q * sqrt(t-1)
    "es_stage_distribution": "uniform",  # uniform over t in {1..T}

    # --- Policy extraction (repo invariant: Beta MEAN, not mode) ---
    "extraction": "mean",

    # --- Independent DP verifier (plan section 4; task phase03) ---
    "verifier": {
        "d_grid_sizes": [51, 101, 201],   # grid-refinement / Richardson check
        "e_grid_size": 201,
        "d_max_margin": 50.0,             # D_max = T*(e_bar + 2q) + margin
        "terminal_integration": "closed_form",  # F_xi(d), never interpolate R(d)
        "quadrature": "deterministic",    # triangular xi; MC only as fallback
        # Certification: state-wise one-step deviation gaps are the PRIMARY
        # certificate (EXP <= sum_t max_d Delta_t(d) is a true upper bound);
        # root-state EXP^UCB is reported alongside.
        "certificate": "delta_gap_sum",
        "epsilon_over_dw": 0.03,          # certification threshold EXP/DW
    },

    # --- Reproducibility ---
    "seed": 42,
    "seed_list": [42, 43, 44, 45, 46],
}


def theoretical_efforts(cfg: Dict[str, Any] = config) -> Dict[str, float]:
    """Closed-form benchmark targets for the configured parameters.

    Args:
        cfg: Configuration dict (defaults to module-level ``config``).

    Returns:
        Dict with g1, g2(0), and the on-path expected stage-2 effort.
    """
    g1 = g1_two_stage(cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"])
    g2_0 = float(
        g2_two_stage(0.0, cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"], cfg["effort_range"][1])
    )
    return {"g1": g1, "g2_at_0": g2_0, "expected_stage2_effort": g1}


def validate(cfg: Dict[str, Any] = config, strict: bool = True):
    """Validate every q in ``q_list`` against the closed-form validity region.

    Runners MUST call this before training (owner decision 2026-07-09:
    no training minute is spent on parameters outside the validity region).

    Args:
        cfg: Configuration dict (defaults to module-level ``config``).
        strict: Raise ``ValueError`` on any failure instead of returning.

    Returns:
        List of ``TwoStageValidation`` reports, one per q in ``q_list``.

    Raises:
        ValueError: If ``strict`` and any q fails the validity checks.
    """
    reports = []
    failures = []
    for q in cfg["q_list"]:
        rep = validate_two_stage_params(
            q=q,
            w_h=cfg["w_h"],
            w_l=cfg["w_l"],
            k=cfg["k"],
            e_bar=cfg["effort_range"][1],
            u_bar=cfg["u_bar"],
        )
        reports.append(rep)
        if not rep.ok:
            failures.append(f"q={q:g}: " + "; ".join(rep.messages))
    if strict and failures:
        raise ValueError(
            "multi-stage config failed validity checks (q_crit="
            f"{q_crit(cfg['w_h'], cfg['w_l'], cfg['k'], cfg['effort_range'][1], cfg['u_bar']):.3f}):\n"
            + "\n".join(failures)
        )
    return reports


if __name__ == "__main__":
    print("Multi-stage canonical config (c(e) = k e^2)")
    print(f"  w_h={config['w_h']}, w_l={config['w_l']}, k={config['k']:.6g}, "
          f"e_bar={config['effort_range'][1]}")
    qc = q_crit(config["w_h"], config["w_l"], config["k"],
                config["effort_range"][1], config["u_bar"])
    print(f"  q_crit = {qc:.3f}  (q_list = {config['q_list']})")
    eff = theoretical_efforts()
    print(f"  targets at q={config['q']}: g1={eff['g1']:.2f}, g2(0)={eff['g2_at_0']:.2f}, "
          f"E[g2]={eff['expected_stage2_effort']:.2f}")
    for rep in validate(strict=False):
        status = "OK  " if rep.ok else "FAIL"
        print(f"  [{status}] q={rep.q:g}: q_crit={rep.q_crit:.2f}, "
              f"g1={rep.g1:.2f}, g2(0)={rep.g2_at_0:.2f}, U_eq={rep.eq_utility:.3f}, "
              f"curv={rep.stage1_curvature:+.2e}, dev1={rep.max_stage1_deviation_gain:+.2e}, "
              f"dev2={rep.max_stage2_deviation_gain:+.2e}, "
              f"dev(e=0)={rep.zero_effort_deviation_gain:+.3f}")
        for m in rep.messages:
            print(f"         {m}")
