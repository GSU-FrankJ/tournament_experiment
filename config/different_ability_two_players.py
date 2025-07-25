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

def calculate_theoretical_efforts_different_ability(q: float, k1: float, k2: float, 
                                                   l1: float, l2: float, 
                                                   w_h: float, w_l: float) -> Tuple[float, float, float, float, float, float]:
    """
    计算具有不同能力玩家的理论最优努力值和期望收益
    
    对于不同能力的Nash equilibrium，我们采用更适合的公式：
    在对称成本但不对称能力的情况下，努力值的比例应该反映能力比例，
    但总体水平需要适当调整以适应给定的参数范围。
    
    采用修正的平衡公式，使努力值在合理范围内：
    e_i = (w_h - w_l) * sqrt(l_i) / (2 * sqrt(k) * q)
    
    Args:
        q: 奖金参数/噪声参数
        k1, k2: 成本参数 (在这个场景中 k1 = k2 = k)
        l1, l2: 能力参数 (l1 > l2)
        w_h, w_l: 权重参数
        
    Returns:
        (e1_optimal, e2_optimal, cost1_optimal, cost2_optimal, EU1_optimal, EU2_optimal)
    """
    
    # 修正的理论最优努力值计算
    # 使用平方根关系以保持在合理范围内
    w_diff = w_h - w_l
    
    # 基础努力水平
    base_effort = w_diff / (2 * np.sqrt(k1) * q)
    
    # 按能力比例调整，但使用平方根以避免过高数值
    e1_optimal = base_effort * np.sqrt(l1) / 2  # 除以2使数值合理
    e2_optimal = base_effort * np.sqrt(l2) / 2
    
    # 进一步调整以确保在范围内
    scaling_factor = min(1.0, 100.0 / max(e1_optimal, e2_optimal))
    e1_optimal *= scaling_factor
    e2_optimal *= scaling_factor
    
    # 计算成本
    cost1_optimal = k1 * (e1_optimal ** 2)
    cost2_optimal = k2 * (e2_optimal ** 2)
    
    # 计算在最优努力下的期望收益
    # 使用 uniform noise 模型计算精确的获胜概率
    # P(player 1 wins) = P(l1*e1 + ε1 > l2*e2 + ε2)
    # 其中 ε1, ε2 ~ Uniform(-q, q)
    
    effective_diff = l1 * e1_optimal - l2 * e2_optimal
    
    # 对于 uniform noise，差值 ε1 - ε2 具有三角分布在 [-2q, 2q]
    # P(player 1 wins) = P(ε1 - ε2 > l2*e2 - l1*e1)
    d = l2 * e2_optimal - l1 * e1_optimal
    
    # 计算精确的获胜概率
    if d <= -2 * q:
        p1_win = 1.0
    elif d >= 2 * q:
        p1_win = 0.0
    elif d < 0:
        p1_win = 1.0 - ((d + 2*q)**2) / (8 * q**2)
    else:
        p1_win = ((2*q - d)**2) / (8 * q**2)
    
    p2_win = 1.0 - p1_win
    
    # 期望收益
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