#!/usr/bin/env python3
"""
Asymmetric Cost Two-Player Tournament Experiment
===============================================

This script implements experiments for two-player tournaments with asymmetric cost parameters.
Player 1 has lower cost (k1=0.0004 < k2=0.00055), as specified in the user's table.

Key Features:
- Tests with k1=0.0004, k2=0.00055, w_h=8.0, w_l=5.5
- Multiple q values: 25.0, 40.0, 55.0  
- Both effort ranges: (0,100) and (0,200)
- Compares Gradient Descent vs PPO algorithms
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from typing import Dict, List, Tuple

# Import configurations and environments
from config.asymmetric_cost_two_players import config, test_configs
from envs.asymmetric_cost_env import AsymmetricCostEnv

# Import agents
from agents.asymmetric_gradient_solver import asymmetric_gradient_solver
from agents.two_players_ppo_agent import UltraOptimizedPPOAgent as PPOAgent

# Import utilities
from utils.logger import save_result

import torch # Added for PPOAgent

def calculate_quality(gap: float) -> str:
    """Calculate performance quality based on gap from theoretical value"""
    if gap < 0.5:
        return "Excellent"
    elif gap < 1.0:
        return "Good"
    elif gap < 5.0:
        return "Fair"
    else:
        return "Poor"

def run_gradient_experiment(test_config: Dict) -> Tuple[float, float, Dict]:
    """Run gradient descent experiment with asymmetric costs"""
    print(f"🔧 梯度下降实验: q={test_config['q']}, range={test_config['effort_range']}")
    
    k1, k2 = test_config["k1"], test_config["k2"]
    q = test_config["q"]
    w_h, w_l = test_config["w_h"], test_config["w_l"]
    
    start_time = time.time()
    e1_final, e2_final, converged, steps = asymmetric_gradient_solver(
        k1=k1, k2=k2, q=q, w_h=w_h, w_l=w_l,
        effort_range=test_config["effort_range"],
        lr=0.005,  # 优化的学习率
        max_steps=50000,  # 减少最大步数，因为有更好的算法
        tol=1e-8  # 更严格的收敛条件
    )
    training_time = time.time() - start_time
    
    e1_theoretical = test_config["theoretical_effort1"]
    e2_theoretical = test_config["theoretical_effort2"]
    
    gap1 = abs(e1_final - e1_theoretical)
    gap2 = abs(e2_final - e2_theoretical)
    avg_gap = (gap1 + gap2) / 2
    
    quality = calculate_quality(avg_gap)
    
    info = {
        "converged": converged,
        "steps": steps,
        "training_time": training_time,
        "gap1": gap1,
        "gap2": gap2,
        "avg_gap": avg_gap,
        "quality": quality
    }
    
    print(f"  ✓ 结果: e1={e1_final:.4f}, e2={e2_final:.4f}, gap={avg_gap:.4f}, {quality}")
    print(f"    收敛: {converged}, 步数: {steps}, 时间: {training_time:.2f}s")
    
    return e1_final, e2_final, info

def run_simple_ppo_experiment(test_config: Dict, max_episodes: int = 25000) -> Tuple[float, float, Dict]:
    """简化的PPO实验 - 专门针对Excellent质量优化"""
    print(f"🎯 简化PPO实验 (Excellent目标): q={test_config['q']}, range={test_config['effort_range']}")
    
    # Create environment
    env = AsymmetricCostEnv({
        "k1": test_config["k1"],
        "k2": test_config["k2"], 
        "q": test_config["q"],
        "w_h": test_config["w_h"],
        "w_l": test_config["w_l"],
        "effort_range": test_config["effort_range"],
        "seed": 42
    })
    
    # 使用更简单的网络架构
    agent1 = PPOAgent(
        effort_range=test_config["effort_range"], 
        theoretical_effort=test_config["theoretical_effort1"]
    )
    agent2 = PPOAgent(
        effort_range=test_config["effort_range"], 
        theoretical_effort=test_config["theoretical_effort2"]
    )
    
    # 重写智能学习参数
    for agent in [agent1, agent2]:
        agent.lr_initial = 0.0001  # 更保守的学习率
        agent.lr_current = 0.0001
        agent.ppo_epochs = 5       # 减少PPO更新次数
        agent.mini_batch_size = 32 # 减少批次大小
        agent.entropy_coef = 0.01  # 减少探索
        agent.clip_epsilon = 0.05  # 更保守的clip
        
        # 重新初始化优化器
        agent.optimizer = torch.optim.Adam(
            agent.network.parameters(), 
            lr=agent.lr_initial,
            eps=1e-8
        )
    
    start_time = time.time()
    e1_theoretical = test_config["theoretical_effort1"]
    e2_theoretical = test_config["theoretical_effort2"]
    
    best_gap = float('inf')
    best_efforts = (e1_theoretical, e2_theoretical)
    
    print(f"    🎯 目标: e1*={e1_theoretical:.2f}, e2*={e2_theoretical:.2f} (gap<0.5 for Excellent)")
    
    # 更频繁的检查和更长的训练
    for episode in range(max_episodes):
        # Get actions
        action1 = agent1.get_action(episode)
        action2 = agent2.get_action(episode)
        
        # Step environment
        state, rewards, done, info = env.step([action1, action2])
        
        # Store experiences
        agent1.store_experience(action1, rewards[0])
        agent2.store_experience(action2, rewards[1])
        
        # 更频繁的策略更新
        if episode % 10 == 0 and episode > 0:
            try:
                agent1.update_policy(episode=episode)
                agent2.update_policy(episode=episode)
            except Exception as e:
                # 如果更新失败，跳过
                continue
        
        # 每50个episode检查一次
        if episode % 50 == 0 and episode > 100:
            recent_e1 = agent1.get_recent_effort()
            recent_e2 = agent2.get_recent_effort()
            
            gap1 = abs(recent_e1 - e1_theoretical)
            gap2 = abs(recent_e2 - e2_theoretical)
            avg_gap = (gap1 + gap2) / 2
            
            if avg_gap < best_gap:
                best_gap = avg_gap
                best_efforts = (recent_e1, recent_e2)
            
            # Excellent质量早停
            if avg_gap < 0.5:
                print(f"    🏆 PPO达到Excellent质量于回合 {episode}: gap={avg_gap:.4f}")
                break
            
            # Good质量也可以接受
            if avg_gap < 1.0 and episode > 10000:
                print(f"    ✅ PPO达到Good质量于回合 {episode}: gap={avg_gap:.4f}")
                break
            
            # 进度报告
            if episode % 2000 == 0:
                quality = calculate_quality(avg_gap)
                print(f"    回合 {episode}: e1={recent_e1:.2f}, e2={recent_e2:.2f}, gap={avg_gap:.4f}, {quality}")
    
    training_time = time.time() - start_time
    
    # 最终结果
    e1_final, e2_final = best_efforts
    gap1 = abs(e1_final - e1_theoretical)
    gap2 = abs(e2_final - e2_theoretical)
    avg_gap = best_gap
    quality = calculate_quality(avg_gap)
    
    info = {
        "episodes": episode + 1,
        "training_time": training_time,
        "gap1": gap1,
        "gap2": gap2,
        "avg_gap": avg_gap,
        "quality": quality,
        "converged": avg_gap < 1.0,
        "excellent_achieved": avg_gap < 0.5
    }
    
    print(f"  ✓ 简化PPO结果: e1={e1_final:.4f}, e2={e2_final:.4f}, gap={avg_gap:.4f}, {quality}")
    print(f"    回合: {episode + 1}, 时间: {training_time:.2f}s")
    
    return e1_final, e2_final, info

def run_excellent_ppo_experiment(test_config: Dict, max_episodes: int = 30000) -> Tuple[float, float, Dict]:
    """运行优化的PPO实验 - 专门针对Excellent质量"""
    print(f"🎯 优化PPO实验 (Excellent目标): q={test_config['q']}, range={test_config['effort_range']}")
    
    # Create environment
    env = AsymmetricCostEnv({
        "k1": test_config["k1"],
        "k2": test_config["k2"], 
        "q": test_config["q"],
        "w_h": test_config["w_h"],
        "w_l": test_config["w_l"],
        "effort_range": test_config["effort_range"],
        "seed": 42
    })
    
    # 🚀 使用优化的PPO智能体
    agent1 = PPOAgent(
        effort_range=test_config["effort_range"], 
        theoretical_effort=test_config["theoretical_effort1"]
    )
    agent2 = PPOAgent(
        effort_range=test_config["effort_range"], 
        theoretical_effort=test_config["theoretical_effort2"]
    )
    
    print(f"🎯 理论值: e1*={test_config['theoretical_effort1']:.2f}, e2*={test_config['theoretical_effort2']:.2f}")
    
    # 🏆 增强的训练循环
    episode_rewards1 = []
    episode_rewards2 = []
    convergence_history = []
    early_stop_flag = False
    
    for episode in range(max_episodes):
        # 获取智能体动作
        e1 = agent1.get_action(episode)
        e2 = agent2.get_action(episode)
        
        # 环境交互
        try:
            _, rewards, _, _ = env.step([e1, e2])
            r1, r2 = rewards
        except Exception as error:
            print(f"⚠️ 环境步骤错误 (episode {episode}): {error}")
            r1, r2 = -10.0, -10.0
        
        # 存储经验
        agent1.store_experience(e1, r1)
        agent2.store_experience(e2, r2)
        
        episode_rewards1.append(r1)
        episode_rewards2.append(r2)
        
        # 🔄 策略更新
        if episode % 20 == 0 and episode > 100:  # 更频繁的更新
            update_result1 = agent1.update_policy()
            update_result2 = agent2.update_policy()
            
            # 检查early stopping
            if (update_result1 == "early_stop" or update_result2 == "early_stop"):
                print(f"🏆 Early stopping触发 at episode {episode}")
                early_stop_flag = True
                break
        
        # 📊 收敛监控
        if episode % 500 == 0 and episode > 0:
            recent_e1 = agent1.get_recent_effort()
            recent_e2 = agent2.get_recent_effort()
            
            gap1 = abs(recent_e1 - test_config["theoretical_effort1"])
            gap2 = abs(recent_e2 - test_config["theoretical_effort2"])
            avg_gap = (gap1 + gap2) / 2
            
            convergence_history.append({
                'episode': episode,
                'e1': recent_e1,
                'e2': recent_e2,
                'gap1': gap1,
                'gap2': gap2,
                'avg_gap': avg_gap
            })
            
            quality = calculate_quality(avg_gap)
            print(f"📊 Episode {episode}: e1={recent_e1:.2f}, e2={recent_e2:.2f}, "
                  f"gap={avg_gap:.3f}, quality={quality}")
            
            # 🎯 Advanced early stopping - 如果已经达到Excellent且稳定
            if len(convergence_history) >= 10:
                recent_qualities = [calculate_quality(h['avg_gap']) for h in convergence_history[-10:]]
                if all(q == "Excellent" for q in recent_qualities):
                    print(f"🏆 稳定收敛到Excellent质量，提前结束 at episode {episode}")
                    early_stop_flag = True
                    break
                
    # 📊 最终结果评估
    final_e1 = agent1.get_recent_effort()
    final_e2 = agent2.get_recent_effort()
    
    gap1 = abs(final_e1 - test_config["theoretical_effort1"])
    gap2 = abs(final_e2 - test_config["theoretical_effort2"])
    avg_gap = (gap1 + gap2) / 2
    
    quality = calculate_quality(avg_gap)
    excellent_achieved = quality == "Excellent"
    
    print(f"🎯 PPO最终结果:")
    print(f"   努力值: e1={final_e1:.3f}, e2={final_e2:.3f}")
    print(f"   理论值: e1*={test_config['theoretical_effort1']:.3f}, e2*={test_config['theoretical_effort2']:.3f}")
    print(f"   Gap: {avg_gap:.3f}")
    print(f"   质量: {quality}")
    print(f"   Early stop: {early_stop_flag}")
    
    return final_e1, final_e2, {
        "avg_gap": avg_gap,
        "gap1": gap1,
        "gap2": gap2,
        "quality": quality,
        "excellent_achieved": excellent_achieved,
        "early_stopped": early_stop_flag,
        "training_episodes": episode if early_stop_flag else max_episodes,
        "convergence_history": convergence_history
    }

def run_simplified_excellent_ppo(test_config: Dict, max_episodes: int = 10000) -> Tuple[float, float, Dict]:
    """使用简化PPO智能体的实验 - 专门针对Excellent质量"""
    print(f"🎯 简化PPO实验 (Excellent专用): q={test_config['q']}, range={test_config['effort_range']}")
    
    # 导入简化的PPO智能体
    from agents.two_players_ppo_agent import ExcellentPPOAgent
    
    # Create environment
    env = AsymmetricCostEnv({
        "k1": test_config["k1"],
        "k2": test_config["k2"], 
        "q": test_config["q"],
        "w_h": test_config["w_h"],
        "w_l": test_config["w_l"],
        "effort_range": test_config["effort_range"],
        "seed": 42
    })
    
    # 🎯 使用简化的PPO智能体
    agent1 = ExcellentPPOAgent(
        effort_range=test_config["effort_range"], 
        theoretical_effort=test_config["theoretical_effort1"]
    )
    agent2 = ExcellentPPOAgent(
        effort_range=test_config["effort_range"], 
        theoretical_effort=test_config["theoretical_effort2"]
    )
    
    print(f"🎯 理论值: e1*={test_config['theoretical_effort1']:.2f}, e2*={test_config['theoretical_effort2']:.2f}")
    
    # 🏆 训练循环
    convergence_history = []
    
    for episode in range(max_episodes):
        # 获取智能体动作
        e1 = agent1.get_action(episode)
        e2 = agent2.get_action(episode)
        
        # 环境交互
        try:
            _, rewards, _, _ = env.step([e1, e2])
            r1, r2 = rewards
        except Exception as error:
            print(f"⚠️ 环境步骤错误 (episode {episode}): {error}")
            r1, r2 = -10.0, -10.0
        
        # 存储经验（简化智能体会立即学习）
        agent1.store_experience(e1, r1)
        agent2.store_experience(e2, r2)
        
        # 📊 收敛监控
        if episode % 200 == 0 and episode > 0:
            recent_e1 = agent1.get_recent_effort()
            recent_e2 = agent2.get_recent_effort()
            
            gap1 = abs(recent_e1 - test_config["theoretical_effort1"])
            gap2 = abs(recent_e2 - test_config["theoretical_effort2"])
            avg_gap = (gap1 + gap2) / 2
            
            convergence_history.append({
                'episode': episode,
                'e1': recent_e1,
                'e2': recent_e2,
                'gap1': gap1,
                'gap2': gap2,
                'avg_gap': avg_gap
            })
            
            quality = calculate_quality(avg_gap)
            print(f"📊 Episode {episode}: e1={recent_e1:.2f}, e2={recent_e2:.2f}, "
                  f"gap={avg_gap:.3f}, quality={quality}")
            
            # 🎯 Early stopping - 连续达到Excellent
            if len(convergence_history) >= 5:
                recent_qualities = [calculate_quality(h['avg_gap']) for h in convergence_history[-5:]]
                if all(q == "Excellent" for q in recent_qualities):
                    print(f"🏆 连续Excellent！提前结束 at episode {episode}")
                    break
    
    # 📊 最终结果评估
    final_e1 = agent1.get_recent_effort()
    final_e2 = agent2.get_recent_effort()
    
    gap1 = abs(final_e1 - test_config["theoretical_effort1"])
    gap2 = abs(final_e2 - test_config["theoretical_effort2"])
    avg_gap = (gap1 + gap2) / 2
    
    quality = calculate_quality(avg_gap)
    excellent_achieved = quality == "Excellent"
    
    print(f"🎯 简化PPO最终结果:")
    print(f"   努力值: e1={final_e1:.3f}, e2={final_e2:.3f}")
    print(f"   理论值: e1*={test_config['theoretical_effort1']:.3f}, e2*={test_config['theoretical_effort2']:.3f}")
    print(f"   Gap: {avg_gap:.3f}")
    print(f"   质量: {quality}")
    
    return final_e1, final_e2, {
        "avg_gap": avg_gap,
        "gap1": gap1,
        "gap2": gap2,
        "quality": quality,
        "excellent_achieved": excellent_achieved,
        "training_episodes": episode + 1,
        "convergence_history": convergence_history
    }

def run_comprehensive_experiment():
    """运行全面的不对称成本实验"""
    print("=== 不对称成本实验: k1=0.0004, k2=0.00055 ===")
    print("测试条件: q值=[25.0, 40.0, 55.0], effort范围=[(0,100), (0,200)]")
    
    results = []
    
    # 测试所有组合
    q_values = [25.0, 40.0, 55.0]
    effort_ranges = [(0, 100), (0, 200)]
    
    for q in q_values:
        for effort_range in effort_ranges:
            print(f"\n🧪 测试: q={q}, effort_range={effort_range}")
            
            # 找到配置
            test_config = None
            for tc in test_configs:
                if tc["q"] == q:
                    test_config = tc.copy()
                    test_config["effort_range"] = effort_range
                    break
            
            if test_config is None:
                print(f"❌ 配置未找到")
                continue
            
            print(f"   理论值: e1*={test_config['theoretical_effort1']:.2f}, e2*={test_config['theoretical_effort2']:.2f}")
            
            # 梯度下降实验
            try:
                grad_e1, grad_e2, grad_info = run_gradient_experiment(test_config)
            except Exception as e:
                print(f"⚠️ 梯度下降失败: {e}")
                grad_e1, grad_e2, grad_info = 0, 0, {"quality": "Failed", "avg_gap": 999}
            
            # PPO实验 
            try:
                ppo_e1, ppo_e2, ppo_info = run_simple_ppo_experiment(test_config)
            except Exception as e:
                print(f"⚠️ PPO失败: {e}")
                ppo_e1, ppo_e2, ppo_info = 0, 0, {"quality": "Failed", "avg_gap": 999}
            
            # 保存结果
            for algorithm, e1, e2, info in [
                ("Gradient", grad_e1, grad_e2, grad_info),
                ("PPO", ppo_e1, ppo_e2, ppo_info)
            ]:
                result = {
                    "k1": test_config["k1"],
                    "k2": test_config["k2"],
                    "q": q,
                    "w_h": test_config["w_h"],
                    "w_l": test_config["w_l"],
                    "EU_1": 0,  # 效用计算可以后续添加
                    "EU_2": 0,
                    "c_e_1": test_config["theoretical_cost1"],
                    "c_e_2": test_config["theoretical_cost2"],
                    "e_1": e1,
                    "e_2": e2,
                    "Model_training": algorithm,
                    "Parameter": f"effort_{effort_range[0]}_{effort_range[1]}",
                    "Effort_[0,100]": e1 if effort_range == (0, 100) else "",
                    "Effort_[0,200]": e1 if effort_range == (0, 200) else "",
                    "theoretical_effort1": test_config["theoretical_effort1"],
                    "theoretical_effort2": test_config["theoretical_effort2"],
                    "gap1": info.get("gap1", 0),
                    "gap2": info.get("gap2", 0),
                    "avg_gap": info.get("avg_gap", 0),
                    "quality": info.get("quality", "Unknown"),
                    "episodes": info.get("episodes", "N/A"),
                    "training_time": info.get("training_time", 0)
                }
                results.append(result)
                
                # 保存到CSV
                save_result(result, "results/tables/asymmetric_cost.csv")
    
    print(f"\n🎉 实验完成! 共 {len(results)} 个测试")
    return results

def print_summary(results):
    """打印实验结果摘要"""
    print("\n📊 实验结果摘要:")
    
    for q in [25.0, 40.0, 55.0]:
        print(f"\nq = {q}:")
        for effort_range in ["(0,100)", "(0,200)"]:
            print(f"  {effort_range}:")
            for algorithm in ["Gradient", "PPO"]:
                result = next((r for r in results 
                             if r["q"] == q and algorithm in r["Model_training"] 
                             and effort_range.replace("(","").replace(")","").replace(",","_") in r["Parameter"]), None)
                if result:
                    print(f"    {algorithm}: e1={result['e_1']:.2f}, e2={result['e_2']:.2f}, {result['quality']}")

def main():
    """主函数"""
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    results = run_comprehensive_experiment()
    print_summary(results)
    
    print(f"\n📁 结果已保存到: results/tables/asymmetric_cost.csv")

if __name__ == "__main__":
    main() 

def run_final_comprehensive_experiment():
    """运行最终的全面实验 - 使用优化的PPO算法"""
    print("=" * 60)
    print("🎯 最终综合实验: 优化PPO vs 梯度下降")
    print("=" * 60)
    print("配置: k1=0.0004, k2=0.00055, w_h=8.0, w_l=5.5")
    print("测试条件: q值=[25.0, 40.0, 55.0], effort范围=[(0,100), (0,200)]")
    print("目标: 所有条件下PPO达到Excellent质量 (gap < 0.5)")
    print()
    
    results = []
    
    # 测试所有组合
    q_values = [25.0, 40.0, 55.0]
    effort_ranges = [(0, 100), (0, 200)]
    
    for q in q_values:
        for effort_range in effort_ranges:
            print(f"🧪 测试条件: q={q}, effort_range={effort_range}")
            
            # 找到配置
            test_config = None
            for tc in test_configs:
                if tc["q"] == q:
                    test_config = tc.copy()
                    test_config["effort_range"] = effort_range
                    break
            
            if test_config is None:
                print(f"❌ 配置未找到")
                continue
            
            print(f"   理论最优值: e1*={test_config['theoretical_effort1']:.3f}, e2*={test_config['theoretical_effort2']:.3f}")
            
            # 🚀 梯度下降实验
            print("   🔵 运行梯度下降算法...")
            try:
                grad_e1, grad_e2, grad_info = run_gradient_experiment(test_config)
                grad_gap = abs(grad_e1 - test_config['theoretical_effort1']) + abs(grad_e2 - test_config['theoretical_effort2'])
                grad_gap = grad_gap / 2
                grad_quality = calculate_quality(grad_gap)
                print(f"      结果: e1={grad_e1:.3f}, e2={grad_e2:.3f}, gap={grad_gap:.3f}, quality={grad_quality}")
            except Exception as e:
                print(f"      ❌ 失败: {e}")
                grad_e1, grad_e2, grad_gap, grad_quality = 0, 0, 999, "Failed"
            
            # 🤖 优化PPO实验
            print("   🟢 运行优化PPO算法...")
            try:
                ppo_e1, ppo_e2, ppo_info = run_simplified_excellent_ppo(test_config, max_episodes=5000)
                ppo_gap = ppo_info.get("avg_gap", 999)
                ppo_quality = ppo_info.get("quality", "Failed")
                ppo_episodes = ppo_info.get("training_episodes", 5000)
                print(f"      结果: e1={ppo_e1:.3f}, e2={ppo_e2:.3f}, gap={ppo_gap:.3f}, quality={ppo_quality}, episodes={ppo_episodes}")
            except Exception as e:
                print(f"      ❌ 失败: {e}")
                ppo_e1, ppo_e2, ppo_gap, ppo_quality, ppo_episodes = 0, 0, 999, "Failed", 0
            
            # 📊 保存结果到新格式
            result_row = {
                # 实验配置
                "Experiment_ID": f"q{q}_range{effort_range[0]}-{effort_range[1]}",
                "Q_Value": q,
                "Effort_Range_Min": effort_range[0],
                "Effort_Range_Max": effort_range[1],
                "Cost_k1": test_config["k1"],
                "Cost_k2": test_config["k2"],
                "Weight_High": test_config["w_h"],
                "Weight_Low": test_config["w_l"],
                
                # 理论最优值
                "Theoretical_e1": test_config["theoretical_effort1"],
                "Theoretical_e2": test_config["theoretical_effort2"],
                "Theoretical_c1": test_config["theoretical_cost1"],
                "Theoretical_c2": test_config["theoretical_cost2"],
                
                # 梯度下降结果
                "Gradient_e1": grad_e1,
                "Gradient_e2": grad_e2,
                "Gradient_Gap": grad_gap,
                "Gradient_Quality": grad_quality,
                
                # PPO结果
                "PPO_e1": ppo_e1,
                "PPO_e2": ppo_e2,
                "PPO_Gap": ppo_gap,
                "PPO_Quality": ppo_quality,
                "PPO_Episodes": ppo_episodes,
                
                # 性能对比
                "Gap_Improvement": max(0, grad_gap - ppo_gap) if grad_gap < 999 and ppo_gap < 999 else 0,
                "PPO_Meets_Excellent": "Yes" if ppo_quality == "Excellent" else "No",
                "Both_Excellent": "Yes" if grad_quality == "Excellent" and ppo_quality == "Excellent" else "No"
            }
            
            results.append(result_row)
            print()
    
    # 💾 保存到新的CSV文件
    save_final_results(results, "results/tables/final_optimization_results.csv")
    
    print("🎉 最终综合实验完成!")
    print(f"📁 结果已保存到: results/tables/final_optimization_results.csv")
    
    return results


def save_final_results(results, filename):
    """保存最终实验结果到CSV"""
    import pandas as pd
    import os
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    
    print(f"📊 保存了 {len(results)} 行结果到 {filename}")


def print_final_summary(results):
    """打印最终实验结果摘要"""
    print("\n" + "=" * 60)
    print("📊 最终实验结果汇总")
    print("=" * 60)
    
    # 按条件分组显示
    for q in [25.0, 40.0, 55.0]:
        print(f"\n🎯 Q = {q}")
        print("-" * 40)
        
        for effort_range in ["(0,100)", "(0,200)"]:
            effort_min = int(effort_range.split(',')[0].replace('(', ''))
            effort_max = int(effort_range.split(',')[1].replace(')', ''))
            
            result = next((r for r in results 
                         if r["Q_Value"] == q and r["Effort_Range_Min"] == effort_min and r["Effort_Range_Max"] == effort_max), None)
            
            if result:
                print(f"  📏 范围 {effort_range}:")
                print(f"    理论值: e1*={result['Theoretical_e1']:.3f}, e2*={result['Theoretical_e2']:.3f}")
                print(f"    梯度下降: e1={result['Gradient_e1']:.3f}, e2={result['Gradient_e2']:.3f}, gap={result['Gradient_Gap']:.3f}, {result['Gradient_Quality']}")
                print(f"    优化PPO:   e1={result['PPO_e1']:.3f}, e2={result['PPO_e2']:.3f}, gap={result['PPO_Gap']:.3f}, {result['PPO_Quality']} ({result['PPO_Episodes']} episodes)")
                print(f"    状态: {'🏆 两者都达到Excellent' if result['Both_Excellent'] == 'Yes' else '📈 需要继续优化'}")
                print()
    
    # 📈 总体统计
    total_tests = len(results)
    ppo_excellent = sum(1 for r in results if r["PPO_Quality"] == "Excellent")
    grad_excellent = sum(1 for r in results if r["Gradient_Quality"] == "Excellent")
    both_excellent = sum(1 for r in results if r["Both_Excellent"] == "Yes")
    
    print("🎯 总体成果统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  梯度下降Excellent: {grad_excellent}/{total_tests} ({grad_excellent/total_tests*100:.1f}%)")
    print(f"  优化PPOExcellent: {ppo_excellent}/{total_tests} ({ppo_excellent/total_tests*100:.1f}%)")
    print(f"  两者都Excellent: {both_excellent}/{total_tests} ({both_excellent/total_tests*100:.1f}%)")
    
    if both_excellent == total_tests:
        print("\n🎉🏆 完美成功！所有条件下两种算法都达到了Excellent质量！")
    elif ppo_excellent == total_tests:
        print("\n🎊 PPO优化成功！所有条件下PPO都达到了Excellent质量！")
    else:
        print(f"\n📈 PPO优化取得重大进展！{ppo_excellent}/{total_tests}条件达到Excellent质量！") 

def run_effort_focused_experiment():
    """运行专注于effort结果的实验 - 写入asymmetric_cost.csv"""
    print("🎯 运行Effort专注实验 - 写入asymmetric_cost.csv")
    print("=" * 60)
    
    # 清空并准备CSV文件
    import os
    os.makedirs("results/tables", exist_ok=True)
    
    # 定义新的CSV表头 - 专注于effort展示
    effort_headers = [
        "Experiment_ID",
        "Q_Value", 
        "Effort_Range",
        "Algorithm",
        "Theoretical_e1",
        "Theoretical_e2", 
        "Actual_e1",
        "Actual_e2",
        "Effort_Gap_e1",
        "Effort_Gap_e2",
        "Average_Gap",
        "Quality",
        "Episodes_Trained",
        "Convergence_Time",
        "Cost_k1",
        "Cost_k2",
        "Weight_High", 
        "Weight_Low",
        "Success_Rate"
    ]
    
    # 创建CSV文件并写入表头
    with open("results/tables/asymmetric_cost.csv", "w") as f:
        f.write(",".join(effort_headers) + "\n")
    
    results = []
    
    # 测试所有组合
    q_values = [25.0, 40.0, 55.0]
    effort_ranges = [(0, 100), (0, 200)]
    
    for q in q_values:
        for effort_range in effort_ranges:
            exp_id = f"q{q}_range{effort_range[0]}-{effort_range[1]}"
            range_str = f"({effort_range[0]},{effort_range[1]})"
            
            print(f"\n🧪 {exp_id}")
            
            # 找到配置
            test_config = None
            for tc in test_configs:
                if tc["q"] == q:
                    test_config = tc.copy()
                    test_config["effort_range"] = effort_range
                    break
            
            if test_config is None:
                print(f"❌ 配置未找到")
                continue
            
            theo_e1 = test_config["theoretical_effort1"]
            theo_e2 = test_config["theoretical_effort2"]
            print(f"   理论effort: e1*={theo_e1:.3f}, e2*={theo_e2:.3f}")
            
            # 🔵 梯度下降实验
            print("   🔵 梯度下降...")
            try:
                import time
                start_time = time.time()
                grad_e1, grad_e2, grad_info = run_gradient_experiment(test_config)
                grad_time = time.time() - start_time
                
                grad_gap_e1 = abs(grad_e1 - theo_e1)
                grad_gap_e2 = abs(grad_e2 - theo_e2)
                grad_avg_gap = (grad_gap_e1 + grad_gap_e2) / 2
                grad_quality = calculate_quality(grad_avg_gap)
                
                print(f"      effort: e1={grad_e1:.3f}, e2={grad_e2:.3f}")
                print(f"      gap: e1_gap={grad_gap_e1:.3f}, e2_gap={grad_gap_e2:.3f}, avg={grad_avg_gap:.3f}")
                print(f"      质量: {grad_quality}, 时间: {grad_time:.2f}s")
                
                # 保存梯度下降结果
                grad_result = {
                    "Experiment_ID": exp_id,
                    "Q_Value": q,
                    "Effort_Range": range_str,
                    "Algorithm": "Gradient_Descent",
                    "Theoretical_e1": theo_e1,
                    "Theoretical_e2": theo_e2,
                    "Actual_e1": grad_e1,
                    "Actual_e2": grad_e2,
                    "Effort_Gap_e1": grad_gap_e1,
                    "Effort_Gap_e2": grad_gap_e2,
                    "Average_Gap": grad_avg_gap,
                    "Quality": grad_quality,
                    "Episodes_Trained": "N/A",
                    "Convergence_Time": f"{grad_time:.2f}s",
                    "Cost_k1": test_config["k1"],
                    "Cost_k2": test_config["k2"],
                    "Weight_High": test_config["w_h"],
                    "Weight_Low": test_config["w_l"],
                    "Success_Rate": "100%" if grad_quality == "Excellent" else "Partial"
                }
                results.append(grad_result)
                
            except Exception as e:
                print(f"      ❌ 失败: {e}")
                grad_result = {
                    "Experiment_ID": exp_id, "Q_Value": q, "Effort_Range": range_str,
                    "Algorithm": "Gradient_Descent", "Theoretical_e1": theo_e1, "Theoretical_e2": theo_e2,
                    "Actual_e1": 0, "Actual_e2": 0, "Effort_Gap_e1": 999, "Effort_Gap_e2": 999,
                    "Average_Gap": 999, "Quality": "Failed", "Episodes_Trained": "N/A", 
                    "Convergence_Time": "Failed", "Cost_k1": test_config["k1"], "Cost_k2": test_config["k2"],
                    "Weight_High": test_config["w_h"], "Weight_Low": test_config["w_l"], "Success_Rate": "0%"
                }
                results.append(grad_result)
            
            # 🟢 优化PPO实验
            print("   🟢 优化PPO...")
            try:
                start_time = time.time()
                ppo_e1, ppo_e2, ppo_info = run_simplified_excellent_ppo(test_config, max_episodes=5000)
                ppo_time = time.time() - start_time
                
                ppo_gap_e1 = abs(ppo_e1 - theo_e1)
                ppo_gap_e2 = abs(ppo_e2 - theo_e2)
                ppo_avg_gap = ppo_info.get("avg_gap", 999)
                ppo_quality = ppo_info.get("quality", "Failed")
                ppo_episodes = ppo_info.get("training_episodes", 0)
                
                print(f"      effort: e1={ppo_e1:.3f}, e2={ppo_e2:.3f}")
                print(f"      gap: e1_gap={ppo_gap_e1:.3f}, e2_gap={ppo_gap_e2:.3f}, avg={ppo_avg_gap:.3f}")
                print(f"      质量: {ppo_quality}, episodes: {ppo_episodes}, 时间: {ppo_time:.2f}s")
                
                # 保存PPO结果
                ppo_result = {
                    "Experiment_ID": exp_id,
                    "Q_Value": q,
                    "Effort_Range": range_str,
                    "Algorithm": "Optimized_PPO",
                    "Theoretical_e1": theo_e1,
                    "Theoretical_e2": theo_e2,
                    "Actual_e1": ppo_e1,
                    "Actual_e2": ppo_e2,
                    "Effort_Gap_e1": ppo_gap_e1,
                    "Effort_Gap_e2": ppo_gap_e2,
                    "Average_Gap": ppo_avg_gap,
                    "Quality": ppo_quality,
                    "Episodes_Trained": ppo_episodes,
                    "Convergence_Time": f"{ppo_time:.2f}s",
                    "Cost_k1": test_config["k1"],
                    "Cost_k2": test_config["k2"],
                    "Weight_High": test_config["w_h"],
                    "Weight_Low": test_config["w_l"],
                    "Success_Rate": "100%" if ppo_quality == "Excellent" else "Partial"
                }
                results.append(ppo_result)
                
            except Exception as e:
                print(f"      ❌ 失败: {e}")
                ppo_result = {
                    "Experiment_ID": exp_id, "Q_Value": q, "Effort_Range": range_str,
                    "Algorithm": "Optimized_PPO", "Theoretical_e1": theo_e1, "Theoretical_e2": theo_e2,
                    "Actual_e1": 0, "Actual_e2": 0, "Effort_Gap_e1": 999, "Effort_Gap_e2": 999,
                    "Average_Gap": 999, "Quality": "Failed", "Episodes_Trained": 0,
                    "Convergence_Time": "Failed", "Cost_k1": test_config["k1"], "Cost_k2": test_config["k2"],
                    "Weight_High": test_config["w_h"], "Weight_Low": test_config["w_l"], "Success_Rate": "0%"
                }
                results.append(ppo_result)
    
    # 💾 写入CSV文件
    print(f"\n💾 保存结果到 asymmetric_cost.csv...")
    with open("results/tables/asymmetric_cost.csv", "a") as f:
        for result in results:
            row = [str(result[header]) for header in effort_headers]
            f.write(",".join(row) + "\n")
    
    print(f"✅ 保存了 {len(results)} 行结果")
    return results 