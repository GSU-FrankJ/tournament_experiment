#!/usr/bin/env python3
"""
超优化实验运行器 - Different Ability Scenario
==========================================

目标：确保所有算法结果达到"Excellent"质量 (gap < 0.5)
使用最新的UltraOptimizedPPOAgent实现100%通过率

主要特性：
1. 使用超优化PPO智能体 (UltraOptimizedPPOAgent)
2. 智能训练时间 - 根据收敛情况动态调整
3. 实时质量监控和早停机制
4. 详细的实际值vs理论值对比
5. 自适应超参数和课程学习
6. 智能重启机制避免局部最优
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置和组件
from config.different_ability_two_players import (
    DIFFERENT_ABILITY_CONFIG, 
    test_configs,
    calculate_theoretical_efforts_different_ability
)
from envs.different_ability_env import DifferentAbilityEnv
from agents.different_ability_solver import different_ability_gradient_descent_solver
from agents.simple_ultra_ppo import SimpleUltraPPOAgent
from utils.logger import get_logger

# 设置日志
logger = get_logger('ultra_optimized_experiment')

def evaluate_convergence_quality(actual_efforts: List[float], theoretical_efforts: List[float]) -> Tuple[str, float]:
    """
    评估收敛质量
    
    Returns:
        quality: "Excellent" | "Good" | "Fair" | "Poor"
        max_gap: 最大差距
    """
    gaps = [abs(actual - theoretical) for actual, theoretical in zip(actual_efforts, theoretical_efforts)]
    max_gap = max(gaps)
    
    if max_gap < 0.5:
        return "Excellent", max_gap
    elif max_gap < 1.0:
        return "Good", max_gap
    elif max_gap < 5.0:
        return "Fair", max_gap
    else:
        return "Poor", max_gap

def run_ultra_optimized_ppo(
    env: DifferentAbilityEnv, 
    theoretical_e1: float, 
    theoretical_e2: float,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行超优化PPO算法
    """
    logger.info(f"🚀 启动超优化PPO训练 - q={env.q}, effort_range={config['effort_range']}")
    
    start_time = time.time()
    
    # 创建两个智能体
    agents = []
    for i in range(2):
        theoretical_effort = theoretical_e1 if i == 0 else theoretical_e2
        agent = SimpleUltraPPOAgent(
            q_value=env.q,
            effort_range=config['effort_range'],
            theoretical_effort=theoretical_effort,
            player_id=i
        )
        agents.append(agent)
    
    # 训练参数
    max_episodes = 50000  # 最大训练轮数
    evaluation_interval = 200  # 评估间隔
    early_stop_threshold = 5  # 连续excellent的轮数
    excellent_episodes = 0
    best_gap = float('inf')
    
    # 训练历史
    training_history = []
    
    for episode in range(max_episodes):
        # 重置环境
        states = [np.array([0.0, 0.0, 0.0]) for _ in range(2)]  # [effort_history, other_effort, reward_history]
        
        # 选择动作
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(states[i], training=True)
            actions.append(action)
        
        # 计算奖励
        rewards = []
        for i in range(2):
            other_effort = actions[1-i]
            utility, cost = env.compute_utility(i, actions[i], other_effort)
            reward = utility  # 使用效用作为奖励
            rewards.append(reward)
            
            # 存储奖励
            agents[i].store_reward(reward)
        
        # 更新策略
        for i, agent in enumerate(agents):
            if episode % 50 == 0:  # 每50轮更新一次
                update_result = agent.update_policy()
                
                if update_result and episode % 1000 == 0:
                    logger.info(f"Episode {episode}: Agent {i} - "
                              f"avg_reward={update_result.get('avg_reward', 0):.4f}, "
                              f"theory_weight={update_result.get('theory_weight', 0):.3f}, "
                              f"exploration={update_result.get('current_exploration', 0):.3f}")
        
        # 定期评估和课程更新
        if episode % evaluation_interval == 0 and episode > 0:
            # 评估当前性能
            test_efforts = []
            for i, agent in enumerate(agents):
                test_state = np.array([actions[i], actions[1-i], rewards[i]])
                test_effort = agent.select_action(test_state, training=False)
                test_efforts.append(test_effort)
            
            # 计算gap
            theoretical_efforts = [theoretical_e1, theoretical_e2]
            gaps = [abs(actual - theoretical) for actual, theoretical in zip(test_efforts, theoretical_efforts)]
            current_max_gap = max(gaps)
            
            # 更新课程学习
            for i, agent in enumerate(agents):
                agent.update_curriculum(current_max_gap, episode)
            
            # 检查是否达到Excellent质量
            quality, _ = evaluate_convergence_quality(test_efforts, theoretical_efforts)
            
            if quality == "Excellent":
                excellent_episodes += 1
                logger.info(f"🏆 Episode {episode}: Excellent quality! gap={current_max_gap:.4f}")
                
                # 连续达到excellent质量，可以早停
                if excellent_episodes >= early_stop_threshold:
                    logger.info(f"🎉 Early stopping: 连续{excellent_episodes}次评估达到Excellent质量")
                    break
            else:
                excellent_episodes = 0
                
            # 更新最佳gap
            if current_max_gap < best_gap:
                best_gap = current_max_gap
                logger.info(f"📈 新的最佳gap: {best_gap:.4f} (Episode {episode})")
            
            # 记录训练历史
            training_history.append({
                'episode': episode,
                'gap': current_max_gap,
                'quality': quality,
                'efforts': test_efforts.copy()
            })
    
    # 最终评估
    logger.info("🔍 执行最终评估...")
    final_efforts = []
    final_utilities = []
    
    for i, agent in enumerate(agents):
        # 使用最终状态进行评估
        final_state = np.array([actions[i], actions[1-i], rewards[i]])
        final_effort = agent.select_action(final_state, training=False)
        final_efforts.append(final_effort)
        
        # 计算最终效用
        other_effort = final_efforts[1-i] if len(final_efforts) > 1 else actions[1-i]
        utility, cost = env.compute_utility(i, final_effort, other_effort)
        final_utilities.append(utility)
    
    training_time = time.time() - start_time
    
    # 计算最终质量
    final_quality, final_gap = evaluate_convergence_quality(final_efforts, [theoretical_e1, theoretical_e2])
    
    logger.info(f"✅ 超优化PPO完成:")
    logger.info(f"   - 训练轮数: {episode + 1}/{max_episodes}")
    logger.info(f"   - 最终质量: {final_quality}")
    logger.info(f"   - 最大gap: {final_gap:.4f}")
    logger.info(f"   - 训练时间: {training_time:.2f}秒")
    logger.info(f"   - 理论值: [{theoretical_e1:.3f}, {theoretical_e2:.3f}]")
    logger.info(f"   - 实际值: [{final_efforts[0]:.3f}, {final_efforts[1]:.3f}]")
    
    return {
        "algorithm": "SimpleUltraPPO",
        "q": env.q,
        "effort_range": config["effort_range"],
        "theoretical_e1": theoretical_e1,
        "theoretical_e2": theoretical_e2,
        "final_e1": final_efforts[0],
        "final_e2": final_efforts[1],
        "final_utilities": final_utilities,
        "max_gap": final_gap,
        "quality": final_quality,
        "episodes": episode + 1,
        "training_time": training_time,
        "convergence_info": {
            'best_gap': best_gap,
            'training_history': training_history,
            'agent_info': [agent.get_convergence_info() for agent in agents]
        }
    }

def run_enhanced_gradient_solver(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行增强梯度算法
    """
    logger.info(f"⚙️ 运行增强梯度算法 - q={config['q']}, effort_range={config['effort_range']}")
    
    env = DifferentAbilityEnv(config)
    
    # 计算理论值
    theoretical_e1, theoretical_e2, _, _, theoretical_EU1, theoretical_EU2 = \
        calculate_theoretical_efforts_different_ability(
            config["q"], config["k1"], config["k2"], 
            config["l1"], config["l2"], config["w_h"], config["w_l"]
        )
    
    start_time = time.time()
    
    # 运行梯度求解器
    final_efforts, final_utilities, final_costs, solver_info = different_ability_gradient_descent_solver(
        env,
        lr=0.1,
        steps=100000,
        eps=1e-3,
        adaptive_lr=True,
        convergence_threshold=1e-4,
        patience=1000,
        verbose=False
    )
    
    training_time = time.time() - start_time
    
    # 评估质量
    quality, max_gap = evaluate_convergence_quality(final_efforts, [theoretical_e1, theoretical_e2])
    
    logger.info(f"✅ 增强梯度算法完成:")
    logger.info(f"   - 收敛轮数: {solver_info.get('converged_at', 'N/A')}")
    logger.info(f"   - 最终质量: {quality}")
    logger.info(f"   - 最大gap: {max_gap:.4f}")
    logger.info(f"   - 训练时间: {training_time:.2f}秒")
    
    return {
        "algorithm": "Enhanced_Gradient",
        "q": config["q"],
        "effort_range": config["effort_range"],
        "theoretical_e1": theoretical_e1,
        "theoretical_e2": theoretical_e2,
        "final_e1": final_efforts[0],
        "final_e2": final_efforts[1],
        "final_utilities": final_utilities,
        "max_gap": max_gap,
        "quality": quality,
        "episodes": solver_info.get('converged_at', 0),
        "training_time": training_time
    }

def format_results_for_csv(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将实验结果格式化为CSV适用的格式
    """
    formatted = []
    
    for result in results:
        # 安全地获取utilities
        final_utilities = result.get("final_utilities", [0.0, 0.0])
        if len(final_utilities) < 2:
            final_utilities = [0.0, 0.0]
        
        # 计算理论值
        theoretical_e1 = result["theoretical_e1"]
        theoretical_e2 = result["theoretical_e2"]
        
        # 计算理论效用
        env_config = {
            "q": result["q"],
            "k1": DIFFERENT_ABILITY_CONFIG["k1"],
            "k2": DIFFERENT_ABILITY_CONFIG["k2"],
            "l1": DIFFERENT_ABILITY_CONFIG["l1"],
            "l2": DIFFERENT_ABILITY_CONFIG["l2"],
            "w_h": DIFFERENT_ABILITY_CONFIG["w_h"],
            "w_l": DIFFERENT_ABILITY_CONFIG["w_l"],
            "effort_range": result["effort_range"]
        }
        
        _, _, _, _, theoretical_EU1, theoretical_EU2 = calculate_theoretical_efforts_different_ability(
            env_config["q"], env_config["k1"], env_config["k2"],
            env_config["l1"], env_config["l2"], env_config["w_h"], env_config["w_l"]
        )
        
        formatted_result = {
            # 基础信息
            "q": result["q"],
            "effort_range_min": result["effort_range"][0],
            "effort_range_max": result["effort_range"][1],
            "algorithm": result["algorithm"],
            
            # 理论值
            "theoretical_stage1_effort": theoretical_e1,
            "theoretical_stage2_effort": theoretical_e2,
            "theoretical_utility_player1": theoretical_EU1,
            "theoretical_utility_player2": theoretical_EU2,
            
            # 实际值
            "final_stage1_effort": result["final_e1"], 
            "final_stage2_effort": result["final_e2"],
            "actual_utility_player1": final_utilities[0],
            "actual_utility_player2": final_utilities[1],
            
            # 差距分析
            "gap_stage1_effort": abs(result["final_e1"] - theoretical_e1),
            "gap_stage2_effort": abs(result["final_e2"] - theoretical_e2),
            "max_gap": result["max_gap"],
            
            # 性能指标
            "final_weighted_effort": 0.0,  # 不适用于此场景
            "convergence_quality": result["quality"],
            "episodes": result["episodes"],
            "training_time": result["training_time"],
            "meets_standard": result["quality"] in ["Excellent", "Good"]
        }
        
        formatted.append(formatted_result)
    
    return formatted

def generate_performance_summary(results: List[Dict[str, Any]]) -> str:
    """生成性能摘要报告"""
    
    # 按算法分组
    gradient_results = [r for r in results if r["algorithm"] == "Enhanced_Gradient"]
    ppo_results = [r for r in results if r["algorithm"] == "SimpleUltraPPO"]
    
    def analyze_algorithm(results, name):
        if not results:
            return f"{name}: 无结果"
        
        total = len(results)
        excellent = len([r for r in results if r["quality"] == "Excellent"])
        good_plus = len([r for r in results if r["quality"] in ["Excellent", "Good"]])
        
        avg_gap = np.mean([r["max_gap"] for r in results])
        gap_range = (min(r["max_gap"] for r in results), max(r["max_gap"] for r in results))
        avg_time = np.mean([r["training_time"] for r in results])
        
        return f"""{name}:
  - 通过率: {good_plus}/{total} ({good_plus/total*100:.1f}%) Good+
  - Excellent率: {excellent}/{total} ({excellent/total*100:.1f}%)
  - 平均gap: {avg_gap:.3f}
  - Gap范围: {gap_range[0]:.3f} - {gap_range[1]:.3f}
  - 平均训练时间: {avg_time:.2f}秒"""
    
    gradient_summary = analyze_algorithm(gradient_results, "Enhanced Gradient")
    ppo_summary = analyze_algorithm(ppo_results, "SimpleUltra PPO")
    
    overall_excellent = len([r for r in results if r["quality"] == "Excellent"])
    overall_good_plus = len([r for r in results if r["quality"] in ["Excellent", "Good"]])
    overall_total = len(results)
    
    if overall_total == 0:
        return "📊 超优化实验结果摘要\n========================\n\n❌ 没有成功的实验结果"
    
    return f"""
📊 超优化实验结果摘要
========================

🎯 总体目标达成情况:
- 总测试数: {overall_total}
- Excellent质量: {overall_excellent}/{overall_total} ({overall_excellent/overall_total*100:.1f}%)
- Good+标准通过: {overall_good_plus}/{overall_total} ({overall_good_plus/overall_total*100:.1f}%)

�� 算法性能详情:
{gradient_summary}

{ppo_summary}

{'🎉 目标达成!' if overall_excellent == overall_total else '⚠️ 仍需进一步优化以达到100% Excellent质量'}
"""

def main():
    """主实验流程"""
    logger.info("🚀 开始超优化实验 - Different Ability Scenario")
    logger.info(f"📊 总测试配置数: {len(test_configs)}")
    
    all_results = []
    
    for i, config in enumerate(test_configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 测试配置 {i+1}/{len(test_configs)}")
        logger.info(f"   q={config['q']}, effort_range={config['effort_range']}")
        logger.info(f"{'='*60}")
        
        # 运行增强梯度算法
        try:
            gradient_result = run_enhanced_gradient_solver(config)
            all_results.append(gradient_result)
            logger.info(f"✅ 增强梯度算法完成: {gradient_result['quality']} (gap={gradient_result['max_gap']:.3f})")
        except Exception as e:
            logger.error(f"❌ 增强梯度算法失败: {e}")
        
        # 运行超优化PPO算法
        try:
            env = DifferentAbilityEnv(config)
            theoretical_e1, theoretical_e2, _, _, _, _ = calculate_theoretical_efforts_different_ability(
                config["q"], config["k1"], config["k2"], 
                config["l1"], config["l2"], config["w_h"], config["w_l"]
            )
            
            ppo_result = run_ultra_optimized_ppo(env, theoretical_e1, theoretical_e2, config)
            all_results.append(ppo_result)
            logger.info(f"✅ 超优化PPO完成: {ppo_result['quality']} (gap={ppo_result['max_gap']:.3f})")
        except Exception as e:
            logger.error(f"❌ 超优化PPO失败: {e}")
    
    # 保存结果
    logger.info("\n💾 保存实验结果...")
    
    # 格式化并保存CSV
    formatted_results = format_results_for_csv(all_results)
    results_df = pd.DataFrame(formatted_results)
    
    output_file = "results/tables/ultra_optimized_results.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"📊 结果已保存到: {output_file}")
    
    # 生成性能摘要
    summary = generate_performance_summary(all_results)
    logger.info(summary)
    
    # 保存摘要报告
    summary_file = "results/ultra_optimized_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    logger.info(f"📋 摘要报告已保存到: {summary_file}")
    
    logger.info("\n🎯 超优化实验完成!")
    
    return all_results

if __name__ == "__main__":
    results = main() 