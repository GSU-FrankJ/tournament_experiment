"""
两个具有不同能力的玩家配置
Single stage: two players with different ability l_1 > l_2, k = k_1 = k_2
Based on table parameters: l1=10, l2=5, k=0.0004

From the table, we can see the expected efforts should be in a reasonable range.
Let me use a more appropriate formula that accounts for the contest structure.
"""

import numpy as np
from typing import Dict, List, Tuple

# 基础配置参数 - 根据表格要求
DIFFERENT_ABILITY_CONFIG = {
    "k": 0.0004,    # 相同的成本参数 k1 = k2 = k
    "k1": 0.0004,   # 玩家1成本参数
    "k2": 0.0004,   # 玩家2成本参数
    "l1": 10,       # 玩家1能力 (较高) - from table
    "l2": 5,        # 玩家2能力 (较低) - from table
    "w_h": 6.5,     # 高权重 - from table
    "w_l": 3.0,     # 低权重 - from table
    "num_players": 2,
    "seed": 42
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
    EU1_optimal = w_l + p1_win * w_diff - cost1_optimal
    EU2_optimal = w_l + p2_win * w_diff - cost2_optimal

    return e1_optimal, e2_optimal, cost1_optimal, cost2_optimal, EU1_optimal, EU2_optimal

# 预计算所有Q值的理论结果 - 根据表格的三个 q 值
Q_VALUES = [25.0, 40.0, 55.0]
theoretical_results = {}

print("=== Different Ability Two Players Theoretical Calculations ===")
print(f"Parameters: l1={DIFFERENT_ABILITY_CONFIG['l1']}, l2={DIFFERENT_ABILITY_CONFIG['l2']}")
print(f"Cost: k1=k2={DIFFERENT_ABILITY_CONFIG['k']}")
print(f"Rewards: w_h={DIFFERENT_ABILITY_CONFIG['w_h']}, w_l={DIFFERENT_ABILITY_CONFIG['w_l']}")
print()

for q in Q_VALUES:
    e1, e2, c1, c2, eu1, eu2 = calculate_theoretical_efforts_different_ability(
        q, DIFFERENT_ABILITY_CONFIG["k1"], DIFFERENT_ABILITY_CONFIG["k2"],
        DIFFERENT_ABILITY_CONFIG["l1"], DIFFERENT_ABILITY_CONFIG["l2"],
        DIFFERENT_ABILITY_CONFIG["w_h"], DIFFERENT_ABILITY_CONFIG["w_l"]
    )
    theoretical_results[q] = {
        "e1": e1, "e2": e2, 
        "c1": c1, "c2": c2,
        "eu1": eu1, "eu2": eu2
    }
    
    print(f"q = {q}:")
    print(f"  e1* = {e1:.2f}, e2* = {e2:.2f}")
    print(f"  c1* = {c1:.2f}, c2* = {c2:.2f}")
    print(f"  EU1* = {eu1:.2f}, EU2* = {eu2:.2f}")
    print()

# 创建测试配置列表 - 包含多个 effort_range
test_configs = []
effort_ranges = [(0, 100), (0, 200)]  # 两种测试范围

for q in Q_VALUES:
    for effort_range in effort_ranges:
        config = {
            "q": q,
            "k": DIFFERENT_ABILITY_CONFIG["k"],
            "k1": DIFFERENT_ABILITY_CONFIG["k1"],
            "k2": DIFFERENT_ABILITY_CONFIG["k2"],
            "l1": DIFFERENT_ABILITY_CONFIG["l1"],
            "l2": DIFFERENT_ABILITY_CONFIG["l2"],
            "w_h": DIFFERENT_ABILITY_CONFIG["w_h"],
            "w_l": DIFFERENT_ABILITY_CONFIG["w_l"],
            "effort_range": effort_range,
            "num_players": DIFFERENT_ABILITY_CONFIG["num_players"],
            "seed": DIFFERENT_ABILITY_CONFIG["seed"],
            "theoretical_effort1": theoretical_results[q]["e1"],
            "theoretical_effort2": theoretical_results[q]["e2"],
            "theoretical_cost1": theoretical_results[q]["c1"],
            "theoretical_cost2": theoretical_results[q]["c2"],
            "theoretical_eu1": theoretical_results[q]["eu1"],
            "theoretical_eu2": theoretical_results[q]["eu2"],
            "theoretical_efforts": [theoretical_results[q]["e1"], theoretical_results[q]["e2"]],
            "theoretical_costs": [theoretical_results[q]["c1"], theoretical_results[q]["c2"]]
        }
        test_configs.append(config)

print(f"Created {len(test_configs)} test configurations")
print(f"Q values: {Q_VALUES}")
print(f"Effort ranges: {effort_ranges}")

# 导出主要配置用于向后兼容
config = test_configs[0]  # 默认使用 q=25, effort_range=(0,100)

if __name__ == "__main__":
    print("=== Different Ability Configuration Summary ===")
    for i, cfg in enumerate(test_configs):
        print(f"Config {i+1}: q={cfg['q']}, range={cfg['effort_range']}")
        print(f"  Theoretical: e1*={cfg['theoretical_effort1']:.2f}, e2*={cfg['theoretical_effort2']:.2f}") 
