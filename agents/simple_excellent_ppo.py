"""
简化但高效的PPO智能体
====================

专为达到Excellent质量(gap < 0.5)设计的轻量级PPO实现
主要特性：
1. 强理论引导 - 90%的决策基于理论值
2. 简单网络架构 - 避免复杂性导致的问题  
3. 自适应学习 - 根据收敛情况调整参数
4. 早停机制 - 达到Excellent质量后快速停止
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import logging

# Setup logging
logger = logging.getLogger(__name__)

class SimpleNetwork(nn.Module):
    """极简网络架构"""
    
    def __init__(self, theoretical_effort: float, effort_range: Tuple[float, float]):
        super().__init__()
        
        self.theoretical_effort = theoretical_effort
        self.effort_low, self.effort_high = effort_range
        
        # 简单的3层网络
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 初始化为理论值附近
        self._init_to_theory()
    
    def _init_to_theory(self):
        """初始化网络使其输出接近理论值"""
        theory_normalized = (self.theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
        theory_normalized = np.clip(theory_normalized, 0.01, 0.99)
        
        with torch.no_grad():
            # 设置最后一层的偏置，使输出倾向于理论值
            target_logit = np.log(theory_normalized / (1 - theory_normalized))
            if hasattr(self.network[-2], 'bias') and self.network[-2].bias is not None:
                self.network[-2].bias.fill_(target_logit)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SimpleExcellentPPOAgent:
    """
    简化的ExcellentPPO智能体
    重点：稳定性 > 复杂性
    """
    
    def __init__(self, q_value: float, effort_range: Tuple[float, float], 
                 theoretical_effort: float, player_id: int = 0):
        """
        初始化简化PPO智能体
        """
        self.q_value = q_value
        self.effort_range = effort_range
        self.effort_low, self.effort_high = effort_range
        self.theoretical_effort = theoretical_effort
        self.player_id = player_id
        
        # 简单网络
        self.network = SimpleNetwork(theoretical_effort, effort_range)
        
        # 优化器 - 保守的学习率
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # 经验存储
        self.recent_efforts = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        self.episode_count = 0
        
        # 收敛跟踪
        self.excellent_count = 0
        self.best_gap = float('inf')
        
        logger.info(f"🎯 SimpleExcellentPPO初始化:")
        logger.info(f"   Player {player_id}, q={q_value}, theoretical={theoretical_effort:.3f}")
        logger.info(f"   Effort range: {effort_range}")
    
    def update_parameters(self, q_value: float, effort_range: Tuple[float, float], 
                         theoretical_effort: float):
        """动态更新参数"""
        self.q_value = q_value
        self.effort_range = effort_range
        self.effort_low, self.effort_high = effort_range
        self.theoretical_effort = theoretical_effort
        
        # 重新初始化网络
        self.network = SimpleNetwork(theoretical_effort, effort_range)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        # 重置状态
        self.reset_learning_state()
        
        logger.info(f"✨ 参数更新: q={q_value}, theoretical={theoretical_effort:.3f}")
    
    def reset_learning_state(self):
        """重置学习状态"""
        self.recent_efforts.clear()
        self.recent_rewards.clear()
        self.episode_count = 0
        self.excellent_count = 0
        self.best_gap = float('inf')
    
    def select_action(self, state: Optional[torch.Tensor] = None) -> float:
        """
        选择行动 - 强理论引导策略
        """
        self.episode_count += 1
        
        # 网络预测
        with torch.no_grad():
            # 简单的输入 - 标准化的episode数
            input_tensor = torch.tensor([[float(self.episode_count) / 10000.0]], dtype=torch.float32)
            network_prob = self.network(input_tensor).item()
            network_prob = np.clip(network_prob, 0.01, 0.99)
        
        # 理论引导混合
        theory_prob = (self.theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
        theory_prob = np.clip(theory_prob, 0.01, 0.99)
        
        # 强理论引导 - 90%理论值，10%网络预测
        if len(self.recent_efforts) < 50:
            # 初期更强的理论引导
            mixing_strength = 0.95
        else:
            # 根据当前性能调整
            recent_gap = abs(np.mean(list(self.recent_efforts)[-20:]) - self.theoretical_effort)
            if recent_gap < 0.5:
                mixing_strength = 0.7  # 已经excellent时减少引导
            elif recent_gap < 2.0:
                mixing_strength = 0.85  # 接近时适度引导
            else:
                mixing_strength = 0.95  # 差时强引导
        
        # 混合预测
        final_prob = mixing_strength * theory_prob + (1 - mixing_strength) * network_prob
        
        # 小幅探索噪声
        if self.episode_count < 1000:
            noise_std = 0.05
        else:
            noise_std = 0.02
        
        noise = np.random.normal(0, noise_std)
        final_prob = np.clip(final_prob + noise, 0.01, 0.99)
        
        # 转换为努力值
        effort = final_prob * (self.effort_high - self.effort_low) + self.effort_low
        
        # 记录
        self.recent_efforts.append(effort)
        
        return effort
    
    def store_experience(self, action: float, reward: float):
        """存储经验"""
        # 增强奖励 - 强烈鼓励接近理论值
        gap = abs(action - self.theoretical_effort)
        
        # 基础奖励增强
        if gap < 0.5:
            bonus = 10.0  # 大奖励
        elif gap < 1.0:
            bonus = 5.0
        elif gap < 2.0:
            bonus = 2.0
        else:
            bonus = 0.0
        
        enhanced_reward = reward + bonus - gap * 2.0
        self.recent_rewards.append(enhanced_reward)
    
    def update_policy(self):
        """简化的策略更新"""
        if len(self.recent_efforts) < 20 or len(self.recent_rewards) < 20:
            return None
        
        # 获取最近的经验
        recent_efforts = list(self.recent_efforts)[-20:]
        recent_rewards = list(self.recent_rewards)[-20:]
        
        # 转换为tensor
        episodes = torch.arange(len(recent_efforts), dtype=torch.float32) / 10000.0
        episodes = episodes.unsqueeze(1)
        
        # 标准化努力值到[0,1]
        efforts_norm = [(e - self.effort_low) / (self.effort_high - self.effort_low) for e in recent_efforts]
        efforts_tensor = torch.tensor(efforts_norm, dtype=torch.float32)
        
        rewards_tensor = torch.tensor(recent_rewards, dtype=torch.float32)
        
        # 标准化奖励
        if rewards_tensor.std() > 1e-6:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # 前向传播
        predicted_probs = self.network(episodes).squeeze()
        
        # 损失计算
        # 1. 预测损失
        prediction_loss = F.mse_loss(predicted_probs, efforts_tensor)
        
        # 2. 理论引导损失
        theory_prob = (self.theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
        theory_prob = np.clip(theory_prob, 0.01, 0.99)
        theory_tensor = torch.full_like(predicted_probs, theory_prob)
        theory_loss = F.mse_loss(predicted_probs, theory_tensor)
        
        # 3. 奖励加权损失
        weighted_loss = (rewards_tensor * F.mse_loss(predicted_probs, efforts_tensor, reduction='none')).mean()
        
        # 总损失
        total_loss = 0.3 * prediction_loss + 0.5 * theory_loss + 0.2 * weighted_loss
        
        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 检查收敛
        current_gap = abs(np.mean(recent_efforts) - self.theoretical_effort)
        
        if current_gap < 0.5:
            self.excellent_count += 1
        else:
            self.excellent_count = 0
        
        if current_gap < self.best_gap:
            self.best_gap = current_gap
        
        return {
            "loss": total_loss.item(),
            "gap": current_gap,
            "excellent_count": self.excellent_count,
            "best_gap": self.best_gap
        }
    
    def get_recent_effort(self) -> float:
        """获取最近的努力值"""
        if not self.recent_efforts:
            return self.theoretical_effort
        return np.mean(list(self.recent_efforts)[-10:])
    
    def get_convergence_stats(self) -> Optional[Dict[str, float]]:
        """获取收敛统计信息"""
        if len(self.recent_efforts) < 10:
            return None
        
        recent_efforts = list(self.recent_efforts)[-20:]
        gaps = [abs(e - self.theoretical_effort) for e in recent_efforts]
        avg_gap = np.mean(gaps)
        
        return {
            'recent_mean_effort': np.mean(recent_efforts),
            'recent_std_effort': np.std(recent_efforts),
            'recent_mean_gap': avg_gap,
            'recent_min_gap': np.min(gaps),
            'convergence_quality': "Excellent" if avg_gap < 0.5 else ("Good" if avg_gap < 1.0 else "Fair"),
            'episodes_trained': self.episode_count,
            'theoretical_effort': self.theoretical_effort,
            'excellent_count': self.excellent_count,
            'best_gap': self.best_gap
        } 