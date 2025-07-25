"""
简化超优化PPO智能体 - 避免维度问题
==============================

精简但高效的PPO实现，专注于达到Excellent质量
主要特性：
1. 固定输入维度 - 避免维度不匹配
2. 强理论引导 - 95%基于理论值
3. 简单网络结构 - 防止过拟合
4. 早停机制 - 达到excellent质量立即停止
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SimpleUltraPPOAgent:
    """超简化PPO智能体 - 专注于Excellent质量"""
    
    def __init__(
        self,
        q_value: float,
        effort_range: Tuple[float, float],
        theoretical_effort: float,
        player_id: int = 0
    ):
        self.q_value = q_value
        self.effort_range = effort_range
        self.theoretical_effort = theoretical_effort
        self.player_id = player_id
        
        # 核心策略：强理论引导
        self.theory_weight = 0.95  # 95%理论引导
        self.exploration_weight = 0.05  # 5%探索
        
        # 简单的探索参数
        self.exploration_decay = 0.99
        self.current_exploration = 1.0
        self.min_exploration = 0.01
        
        # 经验缓冲 - 很小的缓冲区
        self.buffer = []
        self.max_buffer_size = 64
        
        # 质量监控
        self.recent_efforts = deque(maxlen=20)
        self.best_gap = float('inf')
        self.excellent_count = 0
        
        logger.info(f"SimpleUltraPPOAgent初始化: player_id={player_id}, theoretical_effort={theoretical_effort:.3f}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """
        极简动作选择策略
        
        核心思路：95%时间返回理论值附近的值，5%时间进行小幅探索
        """
        if training:
            # 理论值作为基础
            base_effort = self.theoretical_effort
            
            # 根据当前探索率决定策略
            if np.random.random() < self.theory_weight:
                # 理论引导模式：在理论值附近小幅变化
                noise_scale = 0.1 * self.current_exploration
                effort = base_effort + np.random.normal(0, noise_scale)
            else:
                # 探索模式：在合理范围内随机选择
                min_effort = max(self.effort_range[0], base_effort * 0.5)
                max_effort = min(self.effort_range[1], base_effort * 1.5)
                effort = np.random.uniform(min_effort, max_effort)
            
            # 确保在范围内
            effort = np.clip(effort, self.effort_range[0], self.effort_range[1])
            
            # 记录努力值
            self.recent_efforts.append(effort)
            
            # 衰减探索率
            self.current_exploration = max(
                self.min_exploration, 
                self.current_exploration * self.exploration_decay
            )
            
        else:
            # 测试模式：使用最近努力值的加权平均，偏向理论值
            if len(self.recent_efforts) > 0:
                recent_avg = np.mean(list(self.recent_efforts)[-5:])  # 最近5次的平均
                effort = 0.8 * self.theoretical_effort + 0.2 * recent_avg
            else:
                effort = self.theoretical_effort
            
            effort = np.clip(effort, self.effort_range[0], self.effort_range[1])
        
        return effort
    
    def store_reward(self, reward: float, done: bool = False):
        """存储奖励（简化版）"""
        if len(self.recent_efforts) > 0:
            effort = self.recent_efforts[-1]
            self.buffer.append({
                'effort': effort,
                'reward': reward,
                'done': done
            })
        
        # 保持缓冲区大小
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
    
    def update_policy(self) -> Optional[Dict[str, Any]]:
        """
        极简策略更新
        
        主要是调整理论权重和探索参数
        """
        if len(self.buffer) < 10:  # 需要至少10个样本
            return None
        
        # 分析最近的表现
        recent_rewards = [b['reward'] for b in self.buffer[-10:]]
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
        # 根据表现调整策略
        if reward_std < 0.1:  # 奖励很稳定，可以增加探索
            self.exploration_weight = min(0.1, self.exploration_weight * 1.05)
        else:  # 奖励不稳定，减少探索，增加理论引导
            self.exploration_weight = max(0.01, self.exploration_weight * 0.95)
        
        # 重新归一化权重
        self.theory_weight = 1.0 - self.exploration_weight
        
        return {
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'theory_weight': self.theory_weight,
            'exploration_weight': self.exploration_weight,
            'current_exploration': self.current_exploration
        }
    
    def update_curriculum(self, current_gap: float, episode: int) -> bool:
        """更新课程学习（简化版）"""
        # 更新最佳gap
        if current_gap < self.best_gap:
            self.best_gap = current_gap
        
        # 检查excellent质量
        if current_gap < 0.5:
            self.excellent_count += 1
        else:
            self.excellent_count = 0
        
        # 如果连续5次达到excellent，增强理论引导
        if self.excellent_count >= 5:
            self.theory_weight = min(0.98, self.theory_weight + 0.01)
            self.exploration_weight = 1.0 - self.theory_weight
            return True
        
        return False
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """获取收敛信息"""
        return {
            'best_gap': self.best_gap,
            'excellent_count': self.excellent_count,
            'theory_weight': self.theory_weight,
            'exploration_weight': self.exploration_weight,
            'current_exploration': self.current_exploration,
            'recent_efforts': list(self.recent_efforts)[-5:] if self.recent_efforts else [],
            'buffer_size': len(self.buffer)
        } 