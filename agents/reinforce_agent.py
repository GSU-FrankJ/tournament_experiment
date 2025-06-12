import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256):
        super().__init__()
        # 更深的网络架构，提高表达能力
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc5 = nn.Linear(hidden_dim//4, 1)  # output effort mean
        
        # 可学习的log标准差，初始值设置为较小值以减少过度探索
        self.log_std = nn.Parameter(torch.log(torch.ones(1) * 0.5))  # 初始std=0.5
        
        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim//2)
        self.ln4 = nn.LayerNorm(hidden_dim//4)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 智能权重初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化（适合ReLU）
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        
        # 最后一层特殊初始化，输出接近0.5（中等努力水平）
        with torch.no_grad():
            self.fc5.weight.normal_(0, 0.01)
            self.fc5.bias.fill_(0.0)  # sigmoid(0) = 0.5

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.ln4(self.fc4(x)))
        
        mean = torch.sigmoid(self.fc5(x))  # 输出0-1之间的值
        std = torch.exp(self.log_std).expand_as(mean)
        
        # 动态调整标准差，随着训练进行逐渐减小
        return mean, std

class REINFORCEAgent:
    def __init__(self, lr=3e-4, effort_range=(0, 100), log_path=None, baseline_decay=0.99, theoretical_effort=87.5):
        self.policy = PolicyNetwork()
        # 使用AdamW优化器，更好的正则化
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        # 使用cosine退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-6)
        
        self.effort_low, self.effort_high = effort_range
        self.saved_log_probs = []
        self.rewards = []
        self.log_path = log_path
        self.theoretical_effort = theoretical_effort
        
        # 改进的baseline实现
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.baseline_count = 0
        
        # 收敛监控
        self.recent_efforts = []
        self.recent_rewards = []
        self.episode_count = 0
        
        # 自适应探索参数
        self.initial_std = 0.5
        self.min_std = 0.05
        self.std_decay_rate = 0.9995
        
        # 奖励标准化
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []
        
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Effort", "Raw_Reward", "Normalized_Reward", "Loss", "Baseline", "LR", "Std_Dev"])

    def _normalize_reward(self, reward):
        """奖励标准化，提高训练稳定性"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:  # 保持最近1000个奖励
            self.reward_history.pop(0)
        
        if len(self.reward_history) > 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = max(np.std(self.reward_history), 1e-8)
            return (reward - self.reward_mean) / self.reward_std
        return reward

    def select_action(self, state):
        mean, std = self.policy(state)
        
        # 将0-1的输出映射到effort范围
        effort_mean = mean * (self.effort_high - self.effort_low) + self.effort_low
        
        # 自适应标准差衰减
        current_std = max(self.initial_std * (self.std_decay_rate ** self.episode_count), self.min_std)
        
        # 使用自适应标准差
        effort_std = current_std * (self.effort_high - self.effort_low)
        
        # 创建分布并采样
        dist = torch.distributions.Normal(effort_mean, effort_std)
        action = dist.sample()
        
        # 确保动作在合法范围内
        action = torch.clamp(action, self.effort_low, self.effort_high)
        
        # 计算对数概率
        log_prob = dist.log_prob(action)
        
        self.saved_log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        # 标准化奖励
        normalized_reward = self._normalize_reward(reward.item() if torch.is_tensor(reward) else reward)
        self.rewards.append(normalized_reward)

    def update_policy(self, gamma=0.99, episode=None, last_effort=None):
        if not self.rewards:
            return
        
        self.episode_count += 1
        
        # 计算discounted returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 更新baseline（指数移动平均）
        current_return = returns[0].item()
        if self.baseline_count == 0:
            self.baseline = current_return
        else:
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * current_return
        self.baseline_count += 1
        
        # 计算advantages
        advantages = returns - self.baseline
        
        # 标准化advantages（如果有多个step）
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算policy loss
        policy_loss = []
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        if policy_loss:
            # 反向传播
            self.optimizer.zero_grad()
            total_loss = torch.stack(policy_loss).sum()
            total_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 监控收敛
            if last_effort is not None:
                effort_val = last_effort.item() if torch.is_tensor(last_effort) else last_effort
                reward_val = self.rewards[-1]
                
                self.recent_efforts.append(effort_val)
                self.recent_rewards.append(reward_val)
                
                # 保持最近100个结果
                if len(self.recent_efforts) > 100:
                    self.recent_efforts.pop(0)
                    self.recent_rewards.pop(0)

            # 记录训练过程
            if self.log_path and episode is not None and last_effort is not None:
                with open(self.log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    effort_val = last_effort.item() if torch.is_tensor(last_effort) else last_effort
                    raw_reward = self.reward_history[-1] if self.reward_history else 0
                    normalized_reward = self.rewards[-1]
                    current_lr = self.optimizer.param_groups[0]['lr']
                    current_std = max(self.initial_std * (self.std_decay_rate ** self.episode_count), self.min_std)
                    
                    writer.writerow([
                        episode, 
                        round(effort_val, 2), 
                        round(raw_reward, 4),
                        round(normalized_reward, 4),
                        round(total_loss.item(), 4),
                        round(self.baseline, 4),
                        f"{current_lr:.6f}",
                        round(current_std, 4)
                    ])

        # 清理
        self.rewards.clear()
        self.saved_log_probs.clear()
    
    def get_convergence_stats(self):
        """获取收敛统计信息"""
        if len(self.recent_efforts) < 10:
            return None
        
        recent_window = min(50, len(self.recent_efforts))
        recent_mean = np.mean(self.recent_efforts[-recent_window:])
        recent_std = np.std(self.recent_efforts[-recent_window:])
        
        return {
            'recent_mean_effort': recent_mean,
            'recent_std_effort': recent_std,
            'recent_mean_reward': np.mean(self.recent_rewards[-recent_window:])
        }