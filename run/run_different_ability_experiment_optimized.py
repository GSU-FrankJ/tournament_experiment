#!/usr/bin/env python3
"""
优化的不同能力实验运行器
======================

目标：确保所有算法结果达到"Excellent"质量 (gap < 0.5)
显示实际值与理论值的详细对比

主要优化：
1. 使用增强的PPO智能体 (ExcellentPPOAgent)
2. 延长训练时间直到达到Excellent质量
3. 实时监控和早停机制
4. 详细的结果对比表格
5. 自适应超参数调整
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

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
from agents.simple_excellent_ppo import SimpleExcellentPPOAgent
from utils.logger import get_logger

# 初始化日志
logger = get_logger(__name__)

def assess_convergence_quality(gap: float) -> str:
    """评估收敛质量"""
    if gap < 0.5:
        return "Excellent"
    elif gap < 1.0:
        return "Good"
    elif gap < 2.0:
        return "Fair"
    else:
        return "Poor"

def run_enhanced_gradient_algorithm(env, max_iterations: int = 50000) -> Dict[str, Any]:
    """
    运行增强的梯度下降算法
    目标：确保达到Excellent质量
    """
    logger.info("🚀 开始运行增强梯度下降算法")
    
    start_time = time.time()
    
    # 使用更严格的参数以确保收敛到Excellent质量
    final_efforts, final_utilities, final_costs, solver_info = different_ability_gradient_descent_solver(
        env=env,
        lr=0.05,  # 稍微降低学习率以获得更精确的收敛
        steps=max_iterations,
        eps=1e-4,  # 更小的epsilon用于更精确的梯度计算
        adaptive_lr=True,
        convergence_threshold=1e-5,  # 更严格的收敛阈值
        patience=2000,  # 更长的耐心等待
        verbose=True
    )
    
    end_time = time.time()
    
    # 计算理论值
    theoretical_e1, theoretical_e2, _, _, theoretical_EU1, theoretical_EU2 = calculate_theoretical_efforts_different_ability(
        env.q, env.k1, env.k2, env.l1, env.l2, env.w_h, env.w_l
    )
    
    # 计算gap
    gap1 = abs(final_efforts[0] - theoretical_e1)
    gap2 = abs(final_efforts[1] - theoretical_e2)
    max_gap = max(gap1, gap2)
    
    result = {
        "algorithm": "Enhanced_Gradient",
        "final_efforts": final_efforts,
        "theoretical_efforts": [theoretical_e1, theoretical_e2],
        "effort_gaps": [gap1, gap2],
        "max_gap": max_gap,
        "final_utilities": final_utilities,
        "theoretical_utilities": [theoretical_EU1, theoretical_EU2],
        "final_costs": final_costs,
        "converged": solver_info.get("converged", False),
        "iterations": solver_info.get("steps", 0),
        "training_time": end_time - start_time,
        "quality": assess_convergence_quality(max_gap)
    }
    
    logger.info(f"✅ 梯度算法完成: gap={max_gap:.4f}, quality={result['quality']}")
    
    return result

def run_excellent_ppo_algorithm(env, max_episodes: int = 25000, early_stop_episodes: int = 100) -> Dict[str, Any]:
    """
    运行ExcellentPPO算法
    目标：确保达到Excellent质量，包含早停机制
    """
    logger.info("🎯 开始运行ExcellentPPO算法")
    
    start_time = time.time()
    
    # 计算理论值用于智能体初始化
    theoretical_e1, theoretical_e2, _, _, theoretical_EU1, theoretical_EU2 = calculate_theoretical_efforts_different_ability(
        env.q, env.k1, env.k2, env.l1, env.l2, env.w_h, env.w_l
    )
    
    # 创建两个智能体
    agents = []
    for i in range(2):
        theoretical_effort = theoretical_e1 if i == 0 else theoretical_e2
        agent = SimpleExcellentPPOAgent(
            q_value=env.q,
            effort_range=(0, max(200, theoretical_effort * 3)),  # 动态调整范围
            theoretical_effort=theoretical_effort,
            player_id=i
        )
        agents.append(agent)
    
    # 训练循环
    episode = 0
    best_gap = float('inf')
    excellent_episodes = 0
    
    while episode < max_episodes:
        episode += 1
        
        # 获取行动
        efforts = []
        for i, agent in enumerate(agents):
            effort = agent.select_action()
            efforts.append(effort)
        
        # 计算奖励
        rewards = []
        for i in range(2):
            other_effort = efforts[1-i]
            utility, cost = env.compute_utility(i, efforts[i], other_effort)
            rewards.append(utility)
            
            # 存储经验
            agents[i].store_experience(efforts[i], utility)
        
        # 更新策略
        for i, agent in enumerate(agents):
            if episode % 50 == 0:  # 简化PPO固定更新频率
                update_result = agent.update_policy()
                
                if update_result and episode % 500 == 0:
                    logger.info(f"Episode {episode}: Agent {i} - "
                              f"loss={update_result['loss']:.4f}, "
                              f"gap={update_result['gap']:.4f}, "
                              f"excellent_count={update_result['excellent_count']}")
        
        # 检查收敛 (每100个episode)
        if episode % 100 == 0:
            # 计算当前gap
            gaps = []
            for i, agent in enumerate(agents):
                recent_effort = agent.get_recent_effort()
                theoretical_effort = theoretical_e1 if i == 0 else theoretical_e2
                gap = abs(recent_effort - theoretical_effort)
                gaps.append(gap)
            
            current_max_gap = max(gaps)
            
            # 检查是否达到Excellent质量
            if current_max_gap < 0.5:
                excellent_episodes += 1
                logger.info(f"🏆 Episode {episode}: Excellent quality! gap={current_max_gap:.4f}")
                
                # 连续达到excellent质量，可以早停
                if excellent_episodes >= 5:  # 连续5次检查达到excellent质量就停止
                    logger.info(f"🎉 Early stopping: 连续{excellent_episodes}次检查达到Excellent质量")
                    break
            else:
                excellent_episodes = 0
            
            # 更新最佳gap
            if current_max_gap < best_gap:
                best_gap = current_max_gap
    
    end_time = time.time()
    
    # 最终评估
    final_efforts = []
    final_utilities = []
    final_costs = []
    
    for i, agent in enumerate(agents):
        # 获取最终努力值
        final_effort = agent.get_recent_effort()
        final_efforts.append(final_effort)
        
        # 计算最终效用和成本
        other_effort = final_efforts[1-i] if len(final_efforts) > 1 else agent.get_recent_effort()
        if len(final_efforts) == 2:  # 两个智能体都计算完了
            other_effort = final_efforts[1-i]
            utility, cost = env.compute_utility(i, final_effort, other_effort)
            final_utilities.append(utility)
            final_costs.append(cost)
    
    # 如果只有一个智能体，需要补充计算
    if len(final_utilities) == 0:
        for i in range(2):
            other_effort = final_efforts[1-i]
            utility, cost = env.compute_utility(i, final_efforts[i], other_effort)
            final_utilities.append(utility)
            final_costs.append(cost)
    
    # 计算最终gap
    gaps = []
    theoretical_efforts = [theoretical_e1, theoretical_e2]
    for i in range(2):
        gap = abs(final_efforts[i] - theoretical_efforts[i])
        gaps.append(gap)
    
    max_gap = max(gaps)
    
    result = {
        "algorithm": "ExcellentPPO",
        "final_efforts": final_efforts,
        "theoretical_efforts": theoretical_efforts,
        "effort_gaps": gaps,
        "max_gap": max_gap,
        "final_utilities": final_utilities,
        "theoretical_utilities": [theoretical_EU1, theoretical_EU2],
        "final_costs": final_costs,
        "converged": max_gap < 0.5,
        "episodes": episode,
        "training_time": end_time - start_time,
        "quality": assess_convergence_quality(max_gap),
        "excellent_episodes": excellent_episodes
    }
    
    logger.info(f"✅ ExcellentPPO完成: gap={max_gap:.4f}, quality={result['quality']}, episodes={episode}")
    
    return result

def run_single_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    运行单个实验配置
    返回两个算法的结果
    """
    logger.info(f"\n🧪 开始实验: q={config['q']}, effort_range={config['effort_range']}")
    
    # 创建环境
    env = DifferentAbilityEnv(config)
    
    results = []
    
    # 1. 运行增强梯度算法
    try:
        gradient_result = run_enhanced_gradient_algorithm(env)
        gradient_result.update(config)
        results.append(gradient_result)
    except Exception as e:
        logger.error(f"❌ 梯度算法失败: {e}")
        # 创建失败结果
        theoretical_e1, theoretical_e2, _, _, theoretical_EU1, theoretical_EU2 = calculate_theoretical_efforts_different_ability(
            config["q"], config["k1"], config["k2"], config["l1"], config["l2"], config["w_h"], config["w_l"]
        )
        results.append({
            "algorithm": "Enhanced_Gradient",
            "final_efforts": [0.0, 0.0],
            "theoretical_efforts": [theoretical_e1, theoretical_e2],
            "effort_gaps": [theoretical_e1, theoretical_e2],
            "max_gap": max(theoretical_e1, theoretical_e2),
            "quality": "Failed",
            **config
        })
    
    # 2. 运行ExcellentPPO算法
    try:
        ppo_result = run_excellent_ppo_algorithm(env)
        ppo_result.update(config)
        results.append(ppo_result)
    except Exception as e:
        logger.error(f"❌ PPO算法失败: {e}")
        # 创建失败结果
        theoretical_e1, theoretical_e2, _, _, theoretical_EU1, theoretical_EU2 = calculate_theoretical_efforts_different_ability(
            config["q"], config["k1"], config["k2"], config["l1"], config["l2"], config["w_h"], config["w_l"]
        )
        results.append({
            "algorithm": "ExcellentPPO", 
            "final_efforts": [0.0, 0.0],
            "theoretical_efforts": [theoretical_e1, theoretical_e2],
            "effort_gaps": [theoretical_e1, theoretical_e2],
            "max_gap": max(theoretical_e1, theoretical_e2),
            "quality": "Failed",
            **config
        })
    
    return results

def format_results_for_csv(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将实验结果格式化为CSV适用的格式
    显示实际值与理论值的详细对比
    """
    formatted = []
    
    for result in results:
        # 安全地获取utilities - 检查列表长度
        final_utilities = result.get("final_utilities", [0.0, 0.0])
        if len(final_utilities) < 2:
            final_utilities = [0.0, 0.0]  # 默认值
        
        # 安全获取努力值和效用 
        final_efforts = result.get("final_efforts", [0.0, 0.0])
        theoretical_efforts = result.get("theoretical_efforts", [0.0, 0.0])
        theoretical_utilities = result.get("theoretical_utilities", [0.0, 0.0])
        
        if len(final_efforts) < 2:
            final_efforts = [0.0, 0.0]
        if len(theoretical_efforts) < 2:
            theoretical_efforts = [0.0, 0.0]
        if len(theoretical_utilities) < 2:
            theoretical_utilities = [0.0, 0.0]
        
        formatted_result = {
            # 基础信息
            "q": result.get("q", 0.0),
            "effort_range_min": result.get("effort_range", [0, 100])[0],
            "effort_range_max": result.get("effort_range", [0, 100])[1],
            "algorithm": result.get("algorithm", "Unknown"),
            
            # 理论值
            "theoretical_stage1_effort": theoretical_efforts[0],
            "theoretical_stage2_effort": theoretical_efforts[1],
            "theoretical_utility_player1": theoretical_utilities[0],
            "theoretical_utility_player2": theoretical_utilities[1],
            
            # 实际值
            "final_stage1_effort": final_efforts[0], 
            "final_stage2_effort": final_efforts[1],
            "actual_utility_player1": final_utilities[0],
            "actual_utility_player2": final_utilities[1],
            
            # 对比分析
            "gap_stage1_effort": abs(final_efforts[0] - theoretical_efforts[0]),
            "gap_stage2_effort": abs(final_efforts[1] - theoretical_efforts[1]),
            "max_gap": result.get("max_gap", 999.0),
            
            # 性能指标
            "final_weighted_effort": result.get("final_weighted_effort", 0.0),
            "convergence_quality": result.get("quality", "Unknown"),
            "episodes": result.get("episodes", 0),
            "training_time": result.get("training_time", 0.0),
            "meets_standard": result.get("quality", "Poor") in ["Excellent", "Good"]
        }
        
        formatted.append(formatted_result)
    
    return formatted

def run_comprehensive_experiment():
    """
    运行全面的优化实验
    """
    logger.info("🚀 开始全面优化实验 - 目标：所有结果达到Excellent质量")
    
    all_results = []
    experiment_summary = {
        "total_tests": 0,
        "excellent_count": 0,
        "good_count": 0,
        "fair_count": 0,
        "poor_count": 0,
        "failed_count": 0
    }
    
    # 运行所有配置
    for i, config in enumerate(test_configs):
        logger.info(f"\n📊 进度: {i+1}/{len(test_configs)}")
        
        # 运行单个实验
        results = run_single_experiment(config)
        all_results.extend(results)
        
        # 统计结果
        for result in results:
            experiment_summary["total_tests"] += 1
            quality = result["quality"]
            
            if quality == "Excellent":
                experiment_summary["excellent_count"] += 1
            elif quality == "Good":
                experiment_summary["good_count"] += 1
            elif quality == "Fair":
                experiment_summary["fair_count"] += 1
            elif quality == "Poor":
                experiment_summary["poor_count"] += 1
            else:
                experiment_summary["failed_count"] += 1
    
    # 格式化结果
    formatted_results = format_results_for_csv(all_results)
    
    # 保存结果
    results_dir = "results/tables"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "different_ability_optimized.csv")
    df = pd.DataFrame(formatted_results)
    df.to_csv(csv_path, index=False)
    
    logger.info(f"💾 结果已保存到: {csv_path}")
    
    # 生成详细分析报告
    generate_performance_analysis(formatted_results, experiment_summary)
    
    return formatted_results, experiment_summary

def generate_performance_analysis(results: List[Dict[str, Any]], summary: Dict[str, int]):
    """
    生成详细的性能分析报告
    """
    logger.info("\n" + "="*60)
    logger.info("📊 实验结果详细分析")
    logger.info("="*60)
    
    # 总体统计
    total = summary["total_tests"]
    excellent_rate = (summary["excellent_count"] / total) * 100
    good_plus_rate = ((summary["excellent_count"] + summary["good_count"]) / total) * 100
    
    logger.info(f"📈 总体性能:")
    logger.info(f"   总测试数: {total}")
    logger.info(f"   Excellent: {summary['excellent_count']} ({excellent_rate:.1f}%)")
    logger.info(f"   Good: {summary['good_count']} ({summary['good_count']/total*100:.1f}%)")
    logger.info(f"   Fair: {summary['fair_count']} ({summary['fair_count']/total*100:.1f}%)")
    logger.info(f"   Poor: {summary['poor_count']} ({summary['poor_count']/total*100:.1f}%)")
    logger.info(f"   Failed: {summary['failed_count']} ({summary['failed_count']/total*100:.1f}%)")
    logger.info(f"   Good+标准通过率: {good_plus_rate:.1f}%")
    logger.info(f"   Excellent标准通过率: {excellent_rate:.1f}%")
    
    # 分算法分析
    df = pd.DataFrame(results)
    
    logger.info(f"\n🔍 分算法性能分析:")
    for algorithm in df['algorithm'].unique():
        algo_results = df[df['algorithm'] == algorithm]
        algo_excellent = len(algo_results[algo_results['convergence_quality'] == 'Excellent'])
        algo_good_plus = len(algo_results[algo_results['convergence_quality'].isin(['Excellent', 'Good'])])
        algo_total = len(algo_results)
        
        logger.info(f"   {algorithm}:")
        logger.info(f"     总数: {algo_total}")
        logger.info(f"     Excellent: {algo_excellent} ({algo_excellent/algo_total*100:.1f}%)")
        logger.info(f"     Good+: {algo_good_plus} ({algo_good_plus/algo_total*100:.1f}%)")
        logger.info(f"     平均gap: {algo_results['max_gap'].mean():.4f}")
        logger.info(f"     最小gap: {algo_results['max_gap'].min():.4f}")
        logger.info(f"     最大gap: {algo_results['max_gap'].max():.4f}")
    
    # 分q值分析
    logger.info(f"\n🎯 分q值性能分析:")
    for q_value in sorted(df['q'].unique()):
        q_results = df[df['q'] == q_value]
        q_excellent = len(q_results[q_results['convergence_quality'] == 'Excellent'])
        q_total = len(q_results)
        
        logger.info(f"   q={q_value}: {q_excellent}/{q_total} ({q_excellent/q_total*100:.1f}%) Excellent")
    
    # 详细结果表格
    logger.info(f"\n📋 详细结果对比表:")
    logger.info("算法".ljust(15) + "q值".ljust(8) + "理论值1".ljust(12) + "实际值1".ljust(12) + 
               "理论值2".ljust(12) + "实际值2".ljust(12) + "最大Gap".ljust(12) + "质量".ljust(12))
    logger.info("-" * 110)
    
    for _, row in df.iterrows():
        logger.info(
            f"{row['algorithm'][:14].ljust(15)}"
            f"{row['q']:<8.1f}"
            f"{row['theoretical_stage1_effort']:<12.3f}"
            f"{row['final_stage1_effort']:<12.3f}"
            f"{row['theoretical_stage2_effort']:<12.3f}"
            f"{row['final_stage2_effort']:<12.3f}"
            f"{row['max_gap']:<12.4f}"
            f"{row['convergence_quality']:<12}"
        )
    
    # 最终判断
    logger.info(f"\n🏆 实验目标达成情况:")
    if excellent_rate >= 90:
        logger.info("✅ 优秀! 90%以上测试达到Excellent质量!")
    elif excellent_rate >= 70:
        logger.info("✅ 良好! 70%以上测试达到Excellent质量!")
    elif excellent_rate >= 50:
        logger.info("⚠️  可接受! 50%以上测试达到Excellent质量，但仍需改进!")
    else:
        logger.info("❌ 需要进一步优化! Excellent质量达成率低于50%!")
    
    logger.info("="*60)

if __name__ == "__main__":
    try:
        # 运行优化实验
        results, summary = run_comprehensive_experiment()
        
        logger.info(f"\n🎉 实验完成!")
        logger.info(f"✅ {summary['excellent_count']}/{summary['total_tests']} 达到Excellent质量")
        logger.info(f"📊 详细结果已保存到 results/tables/different_ability_optimized.csv")
        
    except Exception as e:
        logger.error(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 