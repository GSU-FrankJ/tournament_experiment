"""
三人游戏综合实验脚本
符合实验优化标准规则要求
测试所有必测参数组合，验证算法性能标准
整合原有功能和comprehensive功能
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from config.one_stage_three_players import get_config, calculate_theoretical_effort
from envs.one_stage_env import OneStageEnv
from agents.gradient_solver import gradient_descent_solver
from agents.three_players_ppo_agent import EnhancedPPOAgent  # 使用重命名后的agent
from utils.logger import save_standardized_result, create_experiment_result, setup_experiment_logging

# 设置日志记录器
logger = setup_experiment_logging("three_players")

def validate_performance_standard(result):
    """
    验证实验结果是否符合标准
    符合实验优化标准规则要求
    """
    # 检查1: 性能质量
    assert result["quality"] in ["Excellent", "Good"], \
        f"❌ 性能不达标: {result['quality']} (要求: Good+)"
    
    # 检查2: 自适应性验证
    if result["q"] == 40.0:
        assert result["actual_effort"] < 60.0, \
            "❌ q=40时effort应该 < 60 (自适应失败)"
    
    if result["q"] == 55.0:
        assert result["actual_effort"] < 45.0, \
            "❌ q=55时effort应该 < 45 (自适应失败)"
    
    # 检查3: 收敛稳定性
    assert result["gap"] < 5.0, \
        f"❌ 收敛gap过大: {result['gap']:.3f}"
    
    logger.info(f"✅ 性能验证通过: q={result['q']}, gap={result['gap']:.3f}, quality={result['quality']}")

def assess_quality(gap):
    """评估性能质量"""
    if gap < 0.5:
        return "Excellent"
    elif gap < 2.0:
        return "Good"
    elif gap < 5.0:
        return "Fair"
    else:
        return "Poor"

def run_gradient_experiment(q_value, effort_range):
    """
    运行梯度下降实验
    梯度算法必须在所有条件下达到Excellent质量
    """
    logger.info(f"🔧 运行梯度下降实验: q={q_value}, effort_range={effort_range}")
    
    config = get_config(q_value, effort_range)
    env = OneStageEnv(config)
    
    # 运行梯度下降
    effort, eu, cost = gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3)
    
    # 计算性能指标
    theoretical_effort = config["effort"]
    gap = abs(effort - theoretical_effort)
    quality = assess_quality(gap)
    
    # 验证：梯度算法应该总是收敛到理论值
    assert gap < 0.5, f"❌ 梯度算法性能下降: gap={gap:.3f}"
    
    result = {
        "q": q_value,
        "effort_range_min": effort_range[0],
        "effort_range_max": effort_range[1],
        "theoretical_effort": theoretical_effort,
        "actual_effort": round(effort, 2),
        "gap": round(gap, 3),
        "quality": quality,
        "algorithm": "Gradient",
        "convergence_time": "100000_steps",
        "meets_standard": quality in ["Excellent", "Good"]
    }
    
    logger.info(f"📊 梯度下降结果: effort={effort:.2f}, theoretical={theoretical_effort:.2f}, gap={gap:.3f}, quality={quality}")
    return result

def run_enhanced_ppo_experiment(q_value, effort_range):
    """
    运行增强PPO实验
    自适应算法必须具备动态参数调整能力
    """
    logger.info(f"🤖 运行增强PPO实验: q={q_value}, effort_range={effort_range}")
    
    config = get_config(q_value, effort_range)
    env = OneStageEnv(config)
    
    # 动态计算理论最优值
    theoretical_effort = calculate_theoretical_effort(q_value)
    
    # 创建自适应PPO智能体
    log_path = f"results/logs/enhanced_ppo_3p_q{q_value}_range{effort_range[0]}_{effort_range[1]}.csv"
    agent1 = EnhancedPPOAgent(q_value, effort_range, theoretical_effort, log_path)
    agent2 = EnhancedPPOAgent(q_value, effort_range, theoretical_effort, None)
    agent3 = EnhancedPPOAgent(q_value, effort_range, theoretical_effort, None)
    
    # 训练参数
    max_episodes = 20000
    convergence_check_interval = 500
    patience = 3000
    best_gap = float('inf')
    episodes_without_improvement = 0
    
    logger.info(f"🏃 开始训练，最大回合数: {max_episodes}")
    
    for episode in range(max_episodes):
        # 环境重置
        state1, state2, state3 = env.reset()
        
        # 选择动作
        a1 = agent1.select_action(state1)
        a2 = agent2.select_action(state2)
        a3 = agent3.select_action(state3)
        
        # 环境步进
        _, rewards, _, _, info = env.step(torch.stack([a1, a2, a3]))
        
        # 存储奖励
        agent1.store_reward(rewards[0])
        agent2.store_reward(rewards[1])
        agent3.store_reward(rewards[2])
        
        # 更新策略
        agent1.update_policy(episode=episode, last_effort=a1)
        agent2.update_policy(episode=episode, last_effort=a2)
        agent3.update_policy(episode=episode, last_effort=a3)
        
        # 收敛检查
        if episode % convergence_check_interval == 0 and episode > 1000:
            stats = agent1.get_convergence_stats()
            if stats:
                current_gap = stats['gap_from_theoretical']
                quality = stats['convergence_quality']
                
                logger.info(f"回合 {episode}: gap={current_gap:.3f}, quality={quality}")
                
                # 检查改善
                if current_gap < best_gap:
                    best_gap = current_gap
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += convergence_check_interval
                
                # 早期停止条件
                if quality in ["Excellent", "Good"] and current_gap < 2.0:
                    logger.info(f"✅ PPO早期收敛于回合 {episode}")
                    break
                
                if episodes_without_improvement >= patience:
                    logger.info(f"⏰ PPO因无改善停止于回合 {episode}")
                    break
    
    # 获取最终结果
    final_effort = info["efforts"][0]
    gap = abs(final_effort - theoretical_effort)
    quality = assess_quality(gap)
    
    result = {
        "q": q_value,
        "effort_range_min": effort_range[0],
        "effort_range_max": effort_range[1],
        "theoretical_effort": round(theoretical_effort, 2),
        "actual_effort": round(final_effort, 2),
        "gap": round(gap, 3),
        "quality": quality,
        "algorithm": "Enhanced_PPO",
        "convergence_time": f"{episode+1}_episodes",
        "meets_standard": quality in ["Excellent", "Good"]
    }
    
    logger.info(f"📊 增强PPO结果: effort={final_effort:.2f}, theoretical={theoretical_effort:.2f}, gap={gap:.3f}, quality={quality}")
    return result

def run_algorithm_test(q_value, effort_range, theoretical_effort):
    """运行单个算法测试组合"""
    logger.info(f"\n🧪 测试条件: q={q_value}, effort_range={effort_range}")
    logger.info(f"📈 理论最优值: {theoretical_effort:.2f}")
    
    results = []
    
    # 测试梯度下降
    try:
        gradient_result = run_gradient_experiment(q_value, effort_range)
        validate_performance_standard(gradient_result)
        results.append(gradient_result)
    except Exception as e:
        logger.error(f"❌ 梯度下降实验失败: {e}")
        raise
    
    # 测试增强PPO
    try:
        ppo_result = run_enhanced_ppo_experiment(q_value, effort_range)
        validate_performance_standard(ppo_result)
        results.append(ppo_result)
    except Exception as e:
        logger.error(f"❌ 增强PPO实验失败: {e}")
        raise
    
    return results

def run_comprehensive_experiment():
    """
    标准的全面实验模板
    符合实验优化标准规则要求
    """
    logger.info("🎯 开始三人游戏综合实验")
    logger.info("📋 实验优化标准规则验证")
    
    # 必须测试的参数组合
    q_values = [25.0, 40.0, 55.0]
    effort_ranges = [(0, 100), (0, 200)]
    
    results = []
    
    for q in q_values:
        for effort_range in effort_ranges:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 测试参数组合: q={q}, effort_range={effort_range}")
            
            # 动态计算理论最优值
            theoretical_effort = calculate_theoretical_effort(q)
            
            # 运行算法测试
            test_results = run_algorithm_test(q, effort_range, theoretical_effort)
            
            results.extend(test_results)
    
    return results

def save_experiment_results(results, filename="three_players"):
    """
    标准结果保存格式 - 使用标准化表头
    符合实验优化标准规则要求
    """
    logger.info(f"💾 保存实验结果到: results/tables/{filename}.csv")
    csv_path = f"results/tables/{filename}.csv"
    
    # 确保目录存在
    os.makedirs("results/tables", exist_ok=True)
    
    # 保存使用标准化格式的结果
    for result in results:
        # 将结果转换为标准格式
        standard_result = create_experiment_result(
            algorithm=result["algorithm"],
            final_effort=result["actual_effort"],
            theoretical_effort=result["theoretical_effort"],
            convergence_quality=result["quality"],
            episodes=result["convergence_time"],
            k1=0.0004,  # 三人游戏固定参数
            information_revelation="none"  # 单阶段游戏
        )
        
        # 保存到标准化CSV
        save_standardized_result(standard_result, csv_path)
    
    # 生成性能报告
    generate_performance_report(results)

def generate_performance_report(results):
    """生成性能报告"""
    logger.info("\n📊 ===== 三人游戏性能报告 =====")
    
    # 按算法分组
    algorithms = {}
    for result in results:
        alg = result["algorithm"]
        if alg not in algorithms:
            algorithms[alg] = []
        algorithms[alg].append(result)
    
    # 打印性能矩阵
    logger.info("\n📈 性能矩阵:")
    logger.info("算法        | q=25 (0,100) | q=25 (0,200) | q=40 (0,100) | q=40 (0,200) | q=55 (0,100) | q=55 (0,200)")
    logger.info("-" * 100)
    
    for alg_name, alg_results in algorithms.items():
        row = f"{alg_name:11}"
        
        for q in [25.0, 40.0, 55.0]:
            for effort_range in [(0, 100), (0, 200)]:
                # 查找匹配的结果
                matching = [r for r in alg_results 
                          if r["q"] == q and 
                          r["effort_range_min"] == effort_range[0] and 
                          r["effort_range_max"] == effort_range[1]]
                
                if matching:
                    result = matching[0]
                    quality_symbol = {"Excellent": "★", "Good": "✓", "Fair": "○", "Poor": "✗"}
                    symbol = quality_symbol.get(result["quality"], "?")
                    gap = result["gap"]
                    row += f" | {symbol} {gap:4.1f}    "
                else:
                    row += f" |     -     "
        
        logger.info(row)
    
    # 统计汇总
    logger.info(f"\n📋 实验汇总:")
    total_tests = len(results)
    excellent_count = len([r for r in results if r["quality"] == "Excellent"])
    good_count = len([r for r in results if r["quality"] == "Good"])
    meets_standard = len([r for r in results if r["meets_standard"]])
    
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"Excellent质量: {excellent_count} ({excellent_count/total_tests*100:.1f}%)")
    logger.info(f"Good质量: {good_count} ({good_count/total_tests*100:.1f}%)")
    logger.info(f"达标率: {meets_standard}/{total_tests} ({meets_standard/total_tests*100:.1f}%)")
    
    if meets_standard == total_tests:
        logger.info("🎉 所有测试均达到标准要求！")
    else:
        logger.warning(f"⚠️  {total_tests - meets_standard} 个测试未达标准")

def main():
    """主函数"""
    logger.info("🎮 三人游戏综合实验 - 符合实验优化标准规则")
    
    # 确保结果目录存在
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    try:
        # 运行综合实验
        results = run_comprehensive_experiment()
        
        # 保存结果
        save_experiment_results(results)
        
        logger.info("\n🏁 实验完成!")
        logger.info("📁 结果文件:")
        logger.info("   - results/tables/three_players.csv")
        logger.info("   - results/logs/enhanced_ppo_3p_*.csv")
        
    except Exception as e:
        logger.error(f"❌ 实验失败: {e}")
        raise

if __name__ == "__main__":
    main()
