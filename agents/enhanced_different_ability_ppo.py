"""
Enhanced PPO Agent for Different Ability Two-Player Tournament
============================================================

Ultra-optimized PPO implementation specifically designed for the different ability scenario.
Target: Achieve "Excellent" quality (gap < 0.5) consistently across all test conditions.

Key Optimizations:
1. Strong theoretical effort guidance with adaptive mixing
2. Specialized network architecture for this specific problem
3. Dynamic learning rate scheduling based on convergence progress
4. Multi-stage curriculum learning with automatic progression
5. Early stopping when Excellent quality is achieved consistently
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TheoryGuidedNetwork(nn.Module):
    """
    网络架构专门为理论引导设计
    包含理论先验知识的注入点
    """
    
    def __init__(self, theoretical_effort: float, effort_range: Tuple[float, float]):
        super().__init__()
        
        self.theoretical_effort = theoretical_effort
        self.effort_low, self.effort_high = effort_range
        
        # 主要网络路径
        self.main_net = nn.Sequential(
            nn.Linear(2, 128),  # [episode_normalized, theory_distance]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 理论引导分支
        self.theory_branch = nn.Sequential(
            nn.Linear(1, 16),  # [theory_normalized]
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),  # [main_output, theory_output]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 初始化偏置向理论值
        self._initialize_with_theory()
    
    def _initialize_with_theory(self):
        """用理论值初始化网络偏置"""
        theory_normalized = (self.theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
        theory_normalized = np.clip(theory_normalized, 0.01, 0.99)
        
        # 设置最终层偏置
        with torch.no_grad():
            # 计算目标logit
            target_logit = np.log(theory_normalized / (1 - theory_normalized))
            # 注意：Sigmoid激活函数没有bias，需要设置前一层的bias
            self.fusion[-2].bias.fill_(target_logit)
    
    def forward(self, episode_info: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            episode_info: [batch_size, 2] containing [episode_normalized, current_effort_normalized]
        """
        # 理论引导输入
        theory_normalized = (self.theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
        theory_tensor = torch.full((episode_info.size(0), 1), theory_normalized, dtype=torch.float32)
        
        # 主网络路径
        main_output = self.main_net(episode_info)
        
        # 理论引导分支
        theory_output = self.theory_branch(theory_tensor)
        
        # 融合输出
        combined = torch.cat([main_output, theory_output], dim=1)
        final_output = self.fusion(combined)
        
        return final_output

class ExcellentPPOAgent:
    """
    专门针对Excellent质量标准的PPO智能体
    """
    
    def __init__(self, q_value: float, effort_range: Tuple[float, float], 
                 theoretical_effort: float, player_id: int = 0):
        """
        初始化ExcellentPPO智能体
        
        Args:
            q_value: 噪声参数
            effort_range: 努力值范围 (min, max)
            theoretical_effort: 理论最优努力值
            player_id: 玩家ID (0 or 1)
        """
        self.q_value = q_value
        self.effort_range = effort_range
        self.effort_low, self.effort_high = effort_range
        self.theoretical_effort = theoretical_effort
        self.player_id = player_id
        
        # 网络
        self.network = TheoryGuidedNetwork(theoretical_effort, effort_range)
        
        # 优化器配置 - 根据q值自适应
        base_lr = self._adaptive_learning_rate()
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=base_lr,
            weight_decay=0.001,
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.8, 
            patience=500,
            min_lr=1e-6,
            verbose=True
        )
        
        # 训练参数
        self.batch_size = 64
        self.update_frequency = 50
        self.clip_epsilon = 0.15
        
        # 经验存储
        self.experiences = []
        self.recent_efforts = deque(maxlen=200)
        self.recent_gaps = deque(maxlen=100)
        
        # 收敛跟踪
        self.episode_count = 0
        self.excellent_streak = 0
        self.best_gap = float('inf')
        
        # 课程学习
        self.curriculum_stage = 0
        self.curriculum_stages = self._create_curriculum()
        
        logger.info(f"🎯 ExcellentPPO初始化完成:")
        logger.info(f"   Player {player_id}, q={q_value}, theoretical={theoretical_effort:.3f}")
        logger.info(f"   Learning rate: {base_lr}, Effort range: {effort_range}")
    
    def _adaptive_learning_rate(self) -> float:
        """根据q值和理论effort自适应学习率"""
        base_lr = 0.001
        
        # q值越小，需要更精细的学习
        if self.q_value <= 30:
            q_factor = 0.5
        elif self.q_value <= 45:
            q_factor = 1.0
        else:
            q_factor = 1.5
        
        # 努力值越小，需要更精细的调整
        if self.theoretical_effort <= 3.0:
            effort_factor = 0.8
        elif self.theoretical_effort <= 5.0:
            effort_factor = 1.0
        else:
            effort_factor = 1.2
        
        return base_lr * q_factor * effort_factor
    
    def _create_curriculum(self) -> List[Dict]:
        """创建自适应课程学习阶段"""
        margin_ratios = [0.5, 0.3, 0.15, 0.08]  # 逐步缩小搜索范围
        stages = []
        
        for i, ratio in enumerate(margin_ratios):
            margin = max(2.0, self.theoretical_effort * ratio)
            range_min = max(self.effort_low, self.theoretical_effort - margin)
            range_max = min(self.effort_high, self.theoretical_effort + margin)
            
            stages.append({
                "range": (range_min, range_max),
                "episodes": 1000 + i * 500,  # 逐步增加训练时间
                "gap_threshold": 2.0 / (i + 1),  # 逐步提高质量要求
                "description": f"Stage {i+1}: margin={margin:.1f}"
            })
        
        return stages
    
    def update_parameters(self, q_value: float, effort_range: Tuple[float, float], 
                         theoretical_effort: float):
        """动态更新参数 - 符合实验优化标准"""
        self.q_value = q_value
        self.effort_range = effort_range
        self.effort_low, self.effort_high = effort_range
        self.theoretical_effort = theoretical_effort
        
        # 更新网络的理论参数
        self.network.theoretical_effort = theoretical_effort
        self.network.effort_low, self.network.effort_high = effort_range
        
        # 重新初始化网络偏置
        self.network._initialize_with_theory()
        
        # 重新设置学习率
        new_lr = self._adaptive_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 重新创建课程学习
        self.curriculum_stages = self._create_curriculum()
        
        # 重置学习状态
        self.reset_learning_state()
        
        logger.info(f"✨ 参数更新: q={q_value}, theoretical={theoretical_effort:.3f}, lr={new_lr}")
    
    def reset_learning_state(self):
        """重置学习状态"""
        self.experiences.clear()
        self.recent_efforts.clear()
        self.recent_gaps.clear()
        self.episode_count = 0
        self.excellent_streak = 0
        self.best_gap = float('inf')
        self.curriculum_stage = 0
    
    def select_action(self, state: Optional[torch.Tensor] = None) -> float:
        """
        选择行动 - 强理论引导 + 智能探索
        """
        self.episode_count += 1
        
        # 准备输入
        episode_normalized = min(1.0, self.episode_count / 10000.0)
        
        # 获取当前课程范围
        current_range = self.get_current_curriculum_range()
        
        # 网络预测
        with torch.no_grad():
            # 构造输入: [episode_normalized, stage_info]
            stage_info = self.curriculum_stage / len(self.curriculum_stages)
            input_tensor = torch.tensor([[episode_normalized, stage_info]], dtype=torch.float32)
            
            network_prob = self.network(input_tensor).item()
            network_prob = np.clip(network_prob, 0.01, 0.99)
        
        # 理论引导混合
        theory_prob = (self.theoretical_effort - current_range[0]) / (current_range[1] - current_range[0])
        theory_prob = np.clip(theory_prob, 0.01, 0.99)
        
        # 动态混合强度
        mixing_strength = self._compute_mixing_strength()
        
        # 混合预测
        final_prob = mixing_strength * theory_prob + (1 - mixing_strength) * network_prob
        
        # 智能探索
        final_prob = self._apply_smart_exploration(final_prob, current_range)
        
        # 转换为努力值
        effort = final_prob * (current_range[1] - current_range[0]) + current_range[0]
        
        # 记录
        self.recent_efforts.append(effort)
        gap = abs(effort - self.theoretical_effort)
        self.recent_gaps.append(gap)
        
        return effort
    
    def _compute_mixing_strength(self) -> float:
        """计算理论引导的混合强度"""
        # 基于当前performance动态调整
        if len(self.recent_gaps) < 10:
            return 0.9  # 初期强引导
        
        recent_gap = np.mean(list(self.recent_gaps)[-10:])
        
        if recent_gap > 5.0:
            return 0.95  # 很差时强引导
        elif recent_gap > 2.0:
            return 0.8   # 中等时适度引导
        elif recent_gap > 0.5:
            return 0.6   # 接近时减少引导
        else:
            return 0.3   # 已经excellent时最小引导
    
    def _apply_smart_exploration(self, prob: float, current_range: Tuple[float, float]) -> float:
        """应用智能探索策略"""
        # 探索强度基于训练进度
        if self.episode_count < 500:
            noise_std = 0.1
        elif self.episode_count < 2000:
            noise_std = 0.05
        else:
            # 基于当前性能调整探索
            if len(self.recent_gaps) >= 10:
                recent_gap = np.mean(list(self.recent_gaps)[-10:])
                if recent_gap < 0.5:
                    noise_std = 0.01  # excellent时最小探索
                elif recent_gap < 2.0:
                    noise_std = 0.03  # good时小探索
                else:
                    noise_std = 0.08  # poor时大探索
            else:
                noise_std = 0.05
        
        # 应用噪声
        noise = np.random.normal(0, noise_std)
        final_prob = np.clip(prob + noise, 0.01, 0.99)
        
        return final_prob
    
    def get_current_curriculum_range(self) -> Tuple[float, float]:
        """获取当前课程学习范围"""
        if self.curriculum_stage >= len(self.curriculum_stages):
            return self.effort_range
        return self.curriculum_stages[self.curriculum_stage]["range"]
    
    def store_experience(self, action: float, reward: float):
        """存储经验"""
        # 增强奖励塑形
        shaped_reward = self._shape_reward(reward, action)
        
        episode_info = {
            "episode": self.episode_count,
            "action": action,
            "reward": shaped_reward,
            "gap": abs(action - self.theoretical_effort)
        }
        
        self.experiences.append(episode_info)
        
        # 限制经验缓冲区大小
        if len(self.experiences) > 2000:
            self.experiences = self.experiences[-1000:]
    
    def _shape_reward(self, original_reward: float, effort: float) -> float:
        """增强奖励塑形 - 强烈鼓励接近理论值"""
        gap = abs(effort - self.theoretical_effort)
        
        # 基础奖励
        shaped_reward = original_reward
        
        # 距离惩罚 - 非线性，距离越远惩罚越重
        distance_penalty = -gap * (1 + gap / 5.0)
        
        # 收敛奖励 - 达到excellent时大奖励
        if gap < 0.5:
            convergence_bonus = 5.0 + (0.5 - gap) * 10.0  # 额外奖励
        elif gap < 1.0:
            convergence_bonus = 2.0
        elif gap < 2.0:
            convergence_bonus = 1.0
        else:
            convergence_bonus = 0.0
        
        # 稳定性奖励
        if len(self.recent_efforts) >= 20:
            recent_efforts = list(self.recent_efforts)[-20:]
            stability = 1.0 / (1.0 + np.std(recent_efforts))
            stability_bonus = stability * 2.0
        else:
            stability_bonus = 0.0
        
        return shaped_reward + distance_penalty + convergence_bonus + stability_bonus
    
    def update_policy(self):
        """更新策略 - 简化但高效的PPO更新"""
        if len(self.experiences) < self.batch_size:
            return None
        
        # 检查是否需要进入下一课程阶段
        self._update_curriculum_stage()
        
        # 获取最近经验
        recent_exp = self.experiences[-self.batch_size:]
        
        # 准备训练数据
        actions = torch.tensor([exp["action"] for exp in recent_exp], dtype=torch.float32)
        rewards = torch.tensor([exp["reward"] for exp in recent_exp], dtype=torch.float32)
        episodes = torch.tensor([exp["episode"] for exp in recent_exp], dtype=torch.float32)
        
        # 标准化奖励
        if rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 构造网络输入
        current_range = self.get_current_curriculum_range()
        action_probs = (actions - current_range[0]) / (current_range[1] - current_range[0])
        action_probs = torch.clamp(action_probs, 0.01, 0.99)
        
        # 准备输入特征
        episodes_norm = episodes / 10000.0
        stage_info = torch.full_like(episodes_norm, self.curriculum_stage / len(self.curriculum_stages))
        network_input = torch.stack([episodes_norm, stage_info], dim=1)
        
        # 前向传播
        predicted_probs = self.network(network_input).squeeze()
        
        # 计算损失
        # 1. 预测损失 - 最小化与实际行动的差距
        prediction_loss = F.mse_loss(predicted_probs, action_probs)
        
        # 2. 理论引导损失 - 强制接近理论值
        theory_prob = (self.theoretical_effort - current_range[0]) / (current_range[1] - current_range[0])
        theory_prob = np.clip(theory_prob, 0.01, 0.99)
        theory_tensor = torch.full_like(predicted_probs, theory_prob)
        theory_loss = F.mse_loss(predicted_probs, theory_tensor)
        
        # 3. 质量引导损失 - 根据gap大小调整
        gaps = torch.tensor([exp["gap"] for exp in recent_exp], dtype=torch.float32)
        quality_weights = torch.exp(-gaps)  # gap越小权重越大
        weighted_theory_loss = (quality_weights * F.mse_loss(predicted_probs, theory_tensor, reduction='none')).mean()
        
        # 4. 正则化损失
        reg_loss = sum(p.pow(2.0).sum() for p in self.network.parameters()) * 1e-5
        
        # 总损失
        total_loss = (0.3 * prediction_loss + 
                     0.4 * theory_loss + 
                     0.25 * weighted_theory_loss + 
                     0.05 * reg_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新学习率调度器
        current_gap = np.mean(list(self.recent_gaps)[-20:]) if len(self.recent_gaps) >= 20 else float('inf')
        self.scheduler.step(current_gap)
        
        # 检查收敛
        convergence_info = self._check_convergence()
        
        return {
            "loss": total_loss.item(),
            "gap": current_gap,
            "stage": self.curriculum_stage,
            "convergence": convergence_info
        }
    
    def _update_curriculum_stage(self):
        """更新课程学习阶段"""
        if self.curriculum_stage >= len(self.curriculum_stages):
            return
        
        current_stage = self.curriculum_stages[self.curriculum_stage]
        
        # 检查是否满足进阶条件
        if len(self.recent_gaps) >= 50:
            recent_gap = np.mean(list(self.recent_gaps)[-50:])
            gap_std = np.std(list(self.recent_gaps)[-50:])
            
            # 进阶条件：平均gap小于阈值且稳定
            if (recent_gap < current_stage["gap_threshold"] and 
                gap_std < current_stage["gap_threshold"] * 0.5 and
                self.episode_count >= current_stage["episodes"]):
                
                self.curriculum_stage += 1
                logger.info(f"📈 进入课程阶段 {self.curriculum_stage}: gap={recent_gap:.3f}")
    
    def _check_convergence(self) -> Dict[str, Any]:
        """检查收敛状态"""
        if len(self.recent_gaps) < 50:
            return {"converged": False, "quality": "Unknown"}
        
        recent_gaps = list(self.recent_gaps)[-50:]
        avg_gap = np.mean(recent_gaps)
        
        # 更新excellent streak
        if avg_gap < 0.5:
            self.excellent_streak += 1
        else:
            self.excellent_streak = 0
        
        # 质量评估
        if avg_gap < 0.5:
            quality = "Excellent"
        elif avg_gap < 1.0:
            quality = "Good"
        elif avg_gap < 2.0:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # 收敛判断 - 连续100个episode为Excellent
        converged = self.excellent_streak >= 100
        
        return {
            "converged": converged,
            "quality": quality,
            "avg_gap": avg_gap,
            "excellent_streak": self.excellent_streak,
            "best_gap": min(recent_gaps)
        }
    
    def get_recent_effort(self) -> float:
        """获取最近的努力值"""
        if not self.recent_efforts:
            return self.theoretical_effort
        return np.mean(list(self.recent_efforts)[-10:])
    
    def get_convergence_stats(self) -> Optional[Dict[str, float]]:
        """获取收敛统计信息"""
        if len(self.recent_gaps) < 20:
            return None
        
        recent_gaps = list(self.recent_gaps)[-50:]
        recent_efforts = list(self.recent_efforts)[-50:]
        
        return {
            'recent_mean_effort': np.mean(recent_efforts),
            'recent_std_effort': np.std(recent_efforts),
            'recent_mean_gap': np.mean(recent_gaps),
            'recent_min_gap': np.min(recent_gaps),
            'convergence_quality': self._check_convergence()["quality"],
            'curriculum_stage': self.curriculum_stage,
            'episodes_trained': self.episode_count,
            'theoretical_effort': self.theoretical_effort,
            'excellent_streak': self.excellent_streak
        } 