"""
两个具有不同能力的玩家配置
Single stage: two players with different ability l_1 > l_2, k = k_1 = k_2
Based on table parameters: l1=10, l2=5.

This module keeps the legacy ``DIFFERENT_ABILITY_CONFIG`` entry point while also
exposing helpers for the parameter grid used in the paper experiments.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Ability parameters shared across experiments
DEFAULT_L1 = 10.0
DEFAULT_L2 = 5.0
DEFAULT_SEED = 42

# Parameters requested in the experiment plan
PARAM_SETS: Tuple[Dict[str, float], ...] = (
    {"k": 0.0004, "w_h": 6.5, "w_l": 3.0},
    {"k": 0.0005, "w_h": 8.0, "w_l": 3.0},
)
Q_VALUES: Tuple[float, ...] = (25.0, 40.0, 55.0)
EFFORT_RANGES: Tuple[Tuple[int, int], ...] = ((0, 100), (0, 200))

PPO_DEFAULTS: Dict[str, object] = {
    # CSV / logging metadata
    "stage1_weight": 3.0,
    "stage2_weight": 6.5,
    "information_revelation": "none",
    "num_players": 2,
    # Training set (mirrors one-stage two-player defaults)
    "q_list": [25.0, 40.0, 55.0],
    "effort_bounds_stage1": [0, 100],
    "effort_bounds_stage2": [0, 200],
    # PPO rollout & optimization
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 3_000_000,
    "max_updates": 0,
    # Opponent (lag) controls
    "opponent_mode": "periodic",
    "opponent_sync_interval": 2,
    "opponent_ema_tau": 0.20,
    "opponent_snapshot_keep": 10,
    "opponent_history_sample_p": 0.0,
    "opponent_history_sample_p_end": 0.0,
    # Lag warm-up / fade schedule
    "lag_warmup_updates": 10,
    "lag_fade_updates": 10,
    # Entropy, LR, clip schedules
    "entropy_coef_start": 0.02,
    "entropy_coef_hold": 0.02,
    "entropy_coef_end": 0.005,
    "entropy_hold_fraction": 2.0 / 3.0,
    "lr_start": 3e-4,
    "lr_end": 2e-4,
    "lr_min": 5e-5,
    "lr_max": 5e-4,
    "clip_range_start": 0.25,
    "clip_range_end": 0.16,
    "clip_range_floor": 0.10,
    "clip_range_ceiling": 0.45,
    "target_kl": 0.010,
    # Evaluation / stopping
    "eval_every_updates": 20,
}


def calculate_theoretical_efforts_different_ability(
    q: float,
    k1: float,
    k2: float,
    l1: float,
    l2: float,
    w_h: float,
    w_l: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    使用解析解计算不同能力下的对称均衡努力与理论效用。

    表中给出的对称均衡努力（e1*=e2*=e*）为：
        e* = ((2q - (l1 - l2)) (w_h - w_l)) / (8 k q^2)

    注意：该模型为加法型能力 y_i = e_i + l_i + ε_i，ε_i ~ U(-q,q)。

    返回：
        (e1*, e2*, c1*, c2*, EU1*, EU2*)
    """
    w_diff = float(w_h - w_l)
    k = float(k1)  # k1 = k2 = k
    delta_l = float(l1 - l2)

    # 解析式（下界到0，避免负努力）
    e_star = ((2.0 * q - delta_l) * w_diff) / (8.0 * k * (q ** 2))
    e_star = max(0.0, e_star)

    e1_optimal = e_star
    e2_optimal = e_star

    # 成本 c(e) = k e^2
    cost1_optimal = k1 * (e1_optimal ** 2)
    cost2_optimal = k2 * (e2_optimal ** 2)

    # 胜率：P(e1 + l1 + ε1 > e2 + l2 + ε2)
    # 令 d = (e2 + l2) - (e1 + l1)，则 ε1 - ε2 ~ Tri(-2q, 2q)
    d = (e2_optimal + l2) - (e1_optimal + l1)
    if d <= -2 * q:
        p1_win = 1.0
    elif d >= 2 * q:
        p1_win = 0.0
    elif d < 0:
        p1_win = 1.0 - ((d + 2 * q) ** 2) / (8 * q ** 2)
    else:
        p1_win = ((2 * q - d) ** 2) / (8 * q ** 2)
    p2_win = 1.0 - p1_win

    # 期望效用
    eu1_optimal = w_l + p1_win * w_diff - cost1_optimal
    eu2_optimal = w_l + p2_win * w_diff - cost2_optimal

    return e1_optimal, e2_optimal, cost1_optimal, cost2_optimal, eu1_optimal, eu2_optimal


def build_different_ability_config(
    *,
    k: float,
    w_h: float,
    w_l: float,
    q: float,
    effort_range: Tuple[int, int],
    l1: float = DEFAULT_L1,
    l2: float = DEFAULT_L2,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Construct a config dict with embedded theoretical quantities."""
    k = float(k)
    k1 = float(k)
    k2 = float(k)
    l1 = float(l1)
    l2 = float(l2)
    w_h = float(w_h)
    w_l = float(w_l)

    e1, e2, c1, c2, eu1, eu2 = calculate_theoretical_efforts_different_ability(
        q, k1, k2, l1, l2, w_h, w_l
    )
    cfg: Dict[str, float] = {
        "k": k,
        "k1": k1,
        "k2": k2,
        "l1": l1,
        "l2": l2,
        "w_h": w_h,
        "w_l": w_l,
        "q": float(q),
        "effort_range": list(effort_range),
        "num_players": 2,
        "seed": int(seed),
        "theoretical_effort1": e1,
        "theoretical_effort2": e2,
        "theoretical_cost1": c1,
        "theoretical_cost2": c2,
        "theoretical_eu1": eu1,
        "theoretical_eu2": eu2,
        "theoretical_efforts": [e1, e2],
        "theoretical_costs": [c1, c2],
    }
    defaults = dict(PPO_DEFAULTS)
    # Always include the current q in q_list for training/eval flexibility
    q_list = list(defaults.get("q_list", []))
    if q not in q_list:
        q_list.append(q)
    defaults["q_list"] = q_list
    # Align effort bounds with the provided range
    low, high = cfg["effort_range"]
    defaults["effort_bounds_stage2"] = [float(low), float(high)]
    defaults["effort_bounds_stage1"] = [float(low), float(min(high, 100.0))]
    cfg.update(defaults)
    return cfg


# Legacy default config (used when no CLI overrides are provided)
DIFFERENT_ABILITY_CONFIG: Dict[str, float] = build_different_ability_config(
    k=PARAM_SETS[0]["k"],
    w_h=PARAM_SETS[0]["w_h"],
    w_l=PARAM_SETS[0]["w_l"],
    q=Q_VALUES[0],
    effort_range=EFFORT_RANGES[0],
)


def build_param_grid_configs() -> List[Dict[str, float]]:
    """Enumerate the full grid (k, w_h, w_l, q, effort_range)."""
    configs: List[Dict[str, float]] = []
    for param_set in PARAM_SETS:
        for q in Q_VALUES:
            for effort_range in EFFORT_RANGES:
                cfg = build_different_ability_config(
                    k=param_set["k"],
                    w_h=param_set["w_h"],
                    w_l=param_set["w_l"],
                    q=q,
                    effort_range=effort_range,
                )
                configs.append(cfg)
    return configs


if __name__ == "__main__":
    print("=== Different Ability Configuration Summary ===")
    for cfg in build_param_grid_configs():
        print(
            f"k={cfg['k']:.4f}, w_h={cfg['w_h']:.1f}, w_l={cfg['w_l']:.1f}, "
            f"q={cfg['q']:.0f}, range={tuple(cfg['effort_range'])}, "
            f"e1*={cfg['theoretical_effort1']:.3f}, e2*={cfg['theoretical_effort2']:.3f}"
        )
