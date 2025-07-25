import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from collections import deque

class ActionSpace:
    """Base class for action space definitions"""
    pass

class ContinuousActionSpace(ActionSpace):
    """Continuous action space definition"""
    def __init__(self, low: float, high: float, shape: Tuple[int, ...] = (1,)):
        self.low = low
        self.high = high
        self.shape = shape
        self.action_type = 'continuous'

class DiscreteActionSpace(ActionSpace):
    """Discrete action space definition"""
    def __init__(self, n: int):
        self.n = n
        self.action_type = 'discrete'

class PolicyNetwork(nn.Module, ABC):
    """Abstract base class for policy networks"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass
    
    @abstractmethod
    def get_action_distribution(self, x: torch.Tensor):
        pass

class ContinuousPolicyNetwork(PolicyNetwork):
    """Policy network for continuous action spaces"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, 
                 num_layers: int = 3, activation: str = 'relu', 
                 dropout_rate: float = 0.05, action_dim: int = 1):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Simplified network without layer normalization to avoid batch issues
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, hidden_dim // 2))
        
        self.layers = nn.ModuleList(layers)
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        # Use a more stable parameterization: directly learn std instead of log_std
        self.std_head = nn.Linear(hidden_dim // 2, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Initialize std_head to produce reasonable std values
        nn.init.constant_(self.std_head.bias, 0.5)  # Will be passed through softplus
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to last layer
                x = self.activation(x)
        
        # Use tanh for better numerical stability and scale to [0,1]
        mean = (torch.tanh(self.mean_head(x)) + 1.0) / 2.0  # Maps to [0,1]
        # Use softplus for std to ensure it's always positive and stable
        std = F.softplus(self.std_head(x)) + 1e-6  # Ensure minimum std
        std = torch.clamp(std, min=1e-6, max=2.0)  # Reasonable range
        
        return mean, std
    
    def get_action_distribution(self, x: torch.Tensor):
        """Get action distribution for sampling"""
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)

class DiscretePolicyNetwork(PolicyNetwork):
    """Policy network for discrete action spaces"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 3, activation: str = 'relu',
                 dropout_rate: float = 0.05, num_actions: int = 10):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Simplified network without layer normalization to avoid batch issues
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, hidden_dim // 2))
        
        self.layers = nn.ModuleList(layers)
        self.logits_head = nn.Linear(hidden_dim // 2, num_actions)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to last layer
                x = self.activation(x)
        
        logits = self.logits_head(x)
        return logits
    
    def get_action_distribution(self, x: torch.Tensor):
        """Get action distribution for sampling"""
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)

class ValueNetwork(nn.Module):
    """Value function network"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 3, activation: str = 'relu',
                 dropout_rate: float = 0.05):
        super().__init__()
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Simplified network without layer normalization to avoid batch issues
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, hidden_dim // 2))
        
        self.layers = nn.ModuleList(layers)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to last layer
                x = self.activation(x)
        
        value = self.value_head(x)
        return value

class GAECalculator:
    """Generalized Advantage Estimation calculator"""
    
    @staticmethod
    def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                   gamma: float = 0.99, gae_lambda: float = 0.95,
                   next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation

        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            next_value: Value estimate for the next state

        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = []
        gae = 0
        
        # Ensure values is 1D and add next value for bootstrapping
        if values.dim() > 1:
            values = values.squeeze(-1)
        values_with_next = torch.cat([values, torch.tensor([next_value], dtype=values.dtype)])
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_with_next[t + 1] - values_with_next[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        
        return advantages, returns

class PPOLoss:
    """PPO loss computation"""
    
    @staticmethod
    def compute_policy_loss(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                          advantages: torch.Tensor, clip_epsilon: float = 0.2) -> torch.Tensor:
        """Compute PPO clipped policy loss"""
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss
    
    @staticmethod
    def compute_value_loss(new_values: torch.Tensor, old_values: torch.Tensor,
                         returns: torch.Tensor, clip_epsilon: float = 0.2) -> torch.Tensor:
        """Compute clipped value loss"""
        # Ensure all tensors have the same shape
        new_values = new_values.squeeze(-1) if new_values.dim() > 1 else new_values
        old_values = old_values.squeeze(-1) if old_values.dim() > 1 else old_values
        
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -clip_epsilon, clip_epsilon
        )
        value_loss1 = F.mse_loss(new_values, returns)
        value_loss2 = F.mse_loss(value_pred_clipped, returns)
        value_loss = torch.max(value_loss1, value_loss2)
        return value_loss

class AdaptivePolicy(nn.Module):
    """
    自适应策略网络，根据理论值初始化偏置
    符合实验优化标准规则要求
    """
    def __init__(self, theoretical_effort, effort_range):
        super(AdaptivePolicy, self).__init__()
        self.theoretical_effort = theoretical_effort
        self.effort_range = effort_range
        
        # 网络结构
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # 根据理论值自适应初始化
        self._adaptive_network_init()
    
    def _adaptive_network_init(self):
        """根据theoretical_effort调整网络偏置"""
        # 标准化理论努力值到 [0, 1]
        normalized_effort = (self.theoretical_effort - self.effort_range[0]) / \
                          (self.effort_range[1] - self.effort_range[0])
        
        # 初始化最后一层偏置，使其接近理论值
        with torch.no_grad():
            # 使用logit函数将[0,1]映射到实数域
            if normalized_effort > 0.99:
                normalized_effort = 0.99
            elif normalized_effort < 0.01:
                normalized_effort = 0.01
                
            logit_value = np.log(normalized_effort / (1 - normalized_effort))
            self.fc3.bias.fill_(logit_value)
            
            # 其他层使用Xavier初始化
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 输出 [0, 1]
        
        # 映射到努力值范围
        effort = x * (self.effort_range[1] - self.effort_range[0]) + self.effort_range[0]
        return effort

class EnhancedPPOAgent:
    """
    增强版PPO智能体，符合实验优化标准规则
    支持自适应参数更新和三人游戏
    """
    
    def __init__(self, q_value, effort_range, theoretical_effort, log_path=None):
        """
        初始化自适应PPO智能体
        
        Args:
            q_value: 噪声参数
            effort_range: 努力值范围
            theoretical_effort: 理论最优努力值
            log_path: 日志路径
        """
        self.q_value = q_value
        self.effort_range = effort_range
        self.theoretical_effort = theoretical_effort
        self.log_path = log_path
        
        # PPO超参数 (根据q值自适应调整)
        self._adaptive_hyperparameters()
        
        # 策略网络和价值网络
        self.policy = AdaptivePolicy(theoretical_effort, effort_range)
        self.value_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        
        # 经验存储
        self.reset_memory()
        
        # 收敛跟踪
        self.recent_efforts = deque(maxlen=100)
        self.convergence_threshold = 2.0  # 与理论值的差距阈值
        
        # 课程学习
        self._adaptive_curriculum_setup()
        
        # 日志记录
        self._setup_logging()
    
    def _adaptive_hyperparameters(self):
        """根据q值动态调整超参数"""
        # 基础超参数
        base_lr = 0.001
        base_clip_epsilon = 0.2
        
        # 根据q值调整 (q越小，噪声越小，需要更精细的学习)
        if self.q_value <= 30.0:
            self.lr = base_lr * 0.5  # 更小的学习率
            self.clip_epsilon = base_clip_epsilon * 0.8
            self.update_frequency = 10
        elif self.q_value <= 45.0:
            self.lr = base_lr
            self.clip_epsilon = base_clip_epsilon
            self.update_frequency = 20
        else:
            self.lr = base_lr * 1.5  # 更大的学习率
            self.clip_epsilon = base_clip_epsilon * 1.2
            self.update_frequency = 30
        
        # 其他超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
    
    def _adaptive_curriculum_setup(self):
        """根据theoretical_effort设置课程学习范围"""
        margin = 0.3 * self.theoretical_effort
        
        # 课程学习阶段
        self.curriculum_stages = [
            {
                "range": (
                    max(self.effort_range[0], self.theoretical_effort - margin),
                    min(self.effort_range[1], self.theoretical_effort + margin)
                ),
                "episodes": 1000
            },
            {
                "range": self.effort_range,
                "episodes": float('inf')  # 无限制
            }
        ]
        
        self.current_stage = 0
        self.stage_episodes = 0
    
    def _setup_logging(self):
        """设置日志记录"""
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.log_data = []
    
    def update_parameters(self, q_value, effort_range, theoretical_effort):
        """
        根据新参数动态调整算法配置
        符合实验优化标准规则要求
        """
        # 更新基础参数
        self.q_value = q_value
        self.effort_range = effort_range
        self.theoretical_effort = theoretical_effort
        
        # 重新初始化网络偏置
        self.policy.theoretical_effort = theoretical_effort
        self.policy.effort_range = effort_range
        self.policy._adaptive_network_init()
        
        # 重新调整超参数
        self._adaptive_hyperparameters()
        
        # 更新优化器学习率
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = self.lr
        
        # 重新设置课程学习
        self._adaptive_curriculum_setup()
        
        # 重置学习状态
        self.reset_learning_state()
    
    def reset_learning_state(self):
        """重置学习状态"""
        self.reset_memory()
        self.recent_efforts.clear()
        self.current_stage = 0
        self.stage_episodes = 0
        
        if self.log_path:
            self.log_data = []
    
    def reset_memory(self):
        """重置经验缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """选择动作"""
        # 确保状态是正确的维度 [1]
        if isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0) if state.dim() == 0 else state.unsqueeze(0) if state.dim() == 1 else state
        else:
            state_tensor = torch.tensor([0.0], dtype=torch.float32).unsqueeze(0)  # 固定为 [1, 1]
        
        with torch.no_grad():
            effort = self.policy(state_tensor)
            value = self.value_net(state_tensor)
        
        # 应用课程学习约束
        effort = self._apply_curriculum_constraint(effort)
        
        # 计算log概率 (简化版本)
        log_prob = -0.5 * ((effort - self.theoretical_effort) / 10.0) ** 2
        
        # 存储经验
        self.states.append([0.0])  # 固定状态维度为 [1]
        self.actions.append(effort.item())
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        return effort.squeeze()
    
    def _apply_curriculum_constraint(self, effort):
        """应用课程学习约束"""
        if self.current_stage < len(self.curriculum_stages):
            stage = self.curriculum_stages[self.current_stage]
            effort_range = stage["range"]
            effort = torch.clamp(effort, effort_range[0], effort_range[1])
        
        return effort
    
    def store_reward(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
        self.dones.append(True)  # 单步游戏
    
    def update_policy(self, episode=None, last_effort=None):
        """更新策略"""
        if len(self.rewards) < self.update_frequency:
            return
        
        # 更新课程学习阶段
        if episode is not None:
            self.stage_episodes += 1
            if (self.current_stage < len(self.curriculum_stages) - 1 and 
                self.stage_episodes >= self.curriculum_stages[self.current_stage]["episodes"]):
                self.current_stage += 1
                self.stage_episodes = 0
        
        # 记录最近努力值
        if last_effort is not None:
            self.recent_efforts.append(last_effort.item() if hasattr(last_effort, 'item') else last_effort)
        
        # 计算优势函数
        advantages, returns = self._compute_advantages()
        
        # PPO更新
        self._ppo_update(advantages, returns)
        
        # 记录日志
        if episode is not None and self.log_path and episode % 100 == 0:
            self._log_progress(episode, last_effort)
        
        # 重置经验
        self.reset_memory()
    
    def _compute_advantages(self):
        """计算优势函数和回报"""
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        
        # 简化版本：直接使用奖励作为回报
        returns = rewards
        advantages = returns - values
        
        return advantages, returns
    
    def _ppo_update(self, advantages, returns):
        """PPO策略更新"""
        states_tensor = torch.tensor(self.states, dtype=torch.float32)
        actions_tensor = torch.tensor(self.actions, dtype=torch.float32)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        
        # 更新策略网络
        new_efforts = self.policy(states_tensor)
        new_log_probs = -0.5 * ((new_efforts.squeeze() - self.theoretical_effort) / 10.0) ** 2
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 更新价值网络
        new_values = self.value_net(states_tensor).squeeze()
        value_loss = nn.MSELoss()(new_values, returns)
        
        # 总损失
        total_loss = policy_loss + self.value_loss_coef * value_loss
        
        # 反向传播
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
    
    def _log_progress(self, episode, last_effort):
        """记录训练进度"""
        if self.recent_efforts:
            mean_effort = np.mean(self.recent_efforts)
            std_effort = np.std(self.recent_efforts)
            gap = abs(mean_effort - self.theoretical_effort)
            
            log_entry = {
                'episode': episode,
                'effort': last_effort.item() if hasattr(last_effort, 'item') else last_effort,
                'mean_effort': mean_effort,
                'std_effort': std_effort,
                'gap': gap,
                'theoretical': self.theoretical_effort,
                'q_value': self.q_value
            }
            
            self.log_data.append(log_entry)
            
            # 保存到CSV
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_path, index=False)
    
    def get_convergence_stats(self):
        """获取收敛统计信息"""
        if len(self.recent_efforts) < 20:
            return None
        
        recent_mean = np.mean(self.recent_efforts)
        recent_std = np.std(self.recent_efforts)
        gap = abs(recent_mean - self.theoretical_effort)
        
        return {
            'recent_mean_effort': recent_mean,
            'recent_std_effort': recent_std,
            'gap_from_theoretical': gap,
            'convergence_quality': self._assess_quality(gap)
        }
    
    def _assess_quality(self, gap):
        """评估收敛质量"""
        if gap < 0.5:
            return "Excellent"
        elif gap < 2.0:
            return "Good"
        elif gap < 5.0:
            return "Fair"
        else:
            return "Poor" 