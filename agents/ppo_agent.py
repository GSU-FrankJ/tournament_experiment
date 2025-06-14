import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=3, activation='relu', dropout_rate=0.05, separate_networks=True):
        super().__init__()
        self.separate_networks = separate_networks
        
        # 可配置的激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.relu
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Policy network - 可配置层数
        policy_layers = []
        current_dim = input_dim
        for i in range(num_layers):
            if i == num_layers - 1:  # 最后一层
                policy_layers.append(nn.Linear(current_dim, hidden_dim//2))
                policy_layers.append(nn.LayerNorm(hidden_dim//2))
            else:
                policy_layers.append(nn.Linear(current_dim, hidden_dim))
                policy_layers.append(nn.LayerNorm(hidden_dim))
                current_dim = hidden_dim
        
        self.policy_layers = nn.ModuleList(policy_layers)
        self.policy_mean = nn.Linear(hidden_dim//2, 1)  # mean for effort
        self.log_std = nn.Parameter(torch.log(torch.ones(1) * 8.0))  # 增加探索
        
        # Value network - 独立或共享网络
        if separate_networks:
            value_layers = []
            current_dim = input_dim
            for i in range(num_layers):
                if i == num_layers - 1:  # 最后一层
                    value_layers.append(nn.Linear(current_dim, hidden_dim//2))
                    value_layers.append(nn.LayerNorm(hidden_dim//2))
                else:
                    value_layers.append(nn.Linear(current_dim, hidden_dim))
                    value_layers.append(nn.LayerNorm(hidden_dim))
                    current_dim = hidden_dim
            
            self.value_layers = nn.ModuleList(value_layers)
        else:
            self.value_layers = self.policy_layers  # 共享网络
        
        self.value_head = nn.Linear(hidden_dim//2, 1)  # state value
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                
        # 特殊初始化策略网络的输出层，使其接近理论值
        with torch.no_grad():
            target_logit = torch.logit(torch.tensor(0.875))  # 理论值/100的logit
            self.policy_mean.bias.fill_(target_logit.item())
            self.policy_mean.weight.fill_(0.01)  # 小权重

    def forward(self, x):
        # Policy network forward pass
        policy_x = x
        for i in range(0, len(self.policy_layers), 2):  # 每两层为一组（Linear + LayerNorm）
            policy_x = self.policy_layers[i](policy_x)  # Linear layer
            if i + 1 < len(self.policy_layers):
                policy_x = self.policy_layers[i + 1](policy_x)  # LayerNorm layer
            policy_x = self.activation(policy_x)
            if i < len(self.policy_layers) - 2:  # 不在最后一层应用dropout
                policy_x = self.dropout(policy_x)
        
        mean = torch.sigmoid(self.policy_mean(policy_x))  # normalized effort [0,1]
        std = torch.exp(self.log_std).expand_as(mean)  # ensure positive std
        
        # Value network forward pass
        value_x = x
        for i in range(0, len(self.value_layers), 2):  # 每两层为一组（Linear + LayerNorm）
            value_x = self.value_layers[i](value_x)  # Linear layer
            if i + 1 < len(self.value_layers):
                value_x = self.value_layers[i + 1](value_x)  # LayerNorm layer
            value_x = self.activation(value_x)
            if i < len(self.value_layers) - 2:  # 不在最后一层应用dropout
                value_x = self.dropout(value_x)
        
        value = self.value_head(value_x)
        
        return mean, std, value

class PPOAgent:
    def __init__(self, lr=1e-4, effort_range=(0, 100), log_path=None, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, theoretical_effort=87.5,
                 hidden_dim=128, num_layers=3, activation='relu', batch_size=64, update_epochs=8, 
                 gae_lambda=0.95, weight_decay=1e-5, dropout_rate=0.05, lr_schedule='constant',
                 separate_networks=True, reward_normalization=True):
        
        self.network = PPONetwork(
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            separate_networks=separate_networks
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay)
        
        # 学习率调度器
        if lr_schedule == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=lr*0.1)
        elif lr_schedule == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        elif lr_schedule == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.9, patience=1500, verbose=True)
        else:
            self.scheduler = None
        
        self.effort_low, self.effort_high = effort_range
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.theoretical_effort = theoretical_effort
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.gae_lambda = gae_lambda
        self.lr_schedule = lr_schedule
        self.reward_normalization = reward_normalization
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
        # 收敛监控
        self.recent_efforts = []
        self.recent_rewards = []
        self.recent_shaped_rewards = []
        
        # 奖励归一化
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []
        
        # 课程学习参数
        self.curriculum_phase = 0
        self.phase_transitions = [8000, 20000]  # 阶段转换点
        
        self.log_path = log_path
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Effort", "Raw_Reward", "Shaped_Reward", "Policy_Loss", "Value_Loss", "Total_Loss", "KL_Div", "Entropy", "LR", "Curriculum_Phase"])

    def _shape_reward(self, raw_reward, effort_value, episode):
        """奖励塑造：引导向理论最优值"""
        effort_val = effort_value.item() if torch.is_tensor(effort_value) else effort_value
        
        # 温和的距离奖励 - 鼓励接近理论值
        distance_to_optimal = abs(effort_val - self.theoretical_effort)
        distance_reward = -distance_to_optimal * 0.01  # 降低强度
        
        # 更温和的课程学习奖励
        if self.curriculum_phase == 0:
            # 第一阶段：鼓励探索高努力水平
            target_range = (40, 90)
            if target_range[0] <= effort_val <= target_range[1]:
                curriculum_reward = 0.5  # 大幅降低奖励强度
            else:
                curriculum_reward = -abs(effort_val - 65) * 0.005
        elif self.curriculum_phase == 1:
            # 第二阶段：鼓励接近理论值
            target_range = (70, 95)
            if target_range[0] <= effort_val <= target_range[1]:
                curriculum_reward = 1.0  # 大幅降低奖励强度
            else:
                curriculum_reward = -abs(effort_val - self.theoretical_effort) * 0.01
        else:
            # 最终阶段：温和鼓励精确值
            if abs(effort_val - self.theoretical_effort) < 5:
                curriculum_reward = 2.0  # 大幅降低奖励强度
            elif abs(effort_val - self.theoretical_effort) < 10:
                curriculum_reward = 1.0
            else:
                curriculum_reward = -abs(effort_val - self.theoretical_effort) * 0.02
        
        # 结合所有奖励，原始奖励为主要成分
        shaped_reward = raw_reward + 0.2 * (distance_reward + curriculum_reward)  # 进一步降低塑造权重
        
        # 根据episode更新课程阶段
        if episode in self.phase_transitions:
            self.curriculum_phase += 1
            print(f"PPO Curriculum learning phase advanced to {self.curriculum_phase}")
        
        return shaped_reward

    def select_action(self, state):
        mean, std, value = self.network(state)
        effort_mean = mean * (self.effort_high - self.effort_low) + self.effort_low
        
        # 移除早期偏置，让优化参数自然工作
        # if len(self.recent_efforts) < 1000:
        #     # 早期阶段，轻微偏向理论值
        #     bias_strength = 0.05
        #     bias = bias_strength * (self.theoretical_effort - effort_mean)
        #     effort_mean = effort_mean + bias
        
        # Create normal distribution and sample
        dist = torch.distributions.Normal(effort_mean, std)
        action = dist.sample()
        action = torch.clamp(action, self.effort_low, self.effort_high)
        log_prob = dist.log_prob(action)
        
        # Store trajectory data (detach to avoid gradient issues)
        self.states.append(state.detach().clone())
        self.actions.append(action.detach().clone())
        self.log_probs.append(log_prob.detach().clone())
        self.values.append(value.detach().clone())
        
        return action

    def store_reward(self, reward):
        reward_val = reward.item() if torch.is_tensor(reward) else reward
        self.rewards.append(reward_val)
        self.reward_history.append(reward_val)
        
        # 保持最近1000个奖励用于归一化
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        # 更新奖励统计
        if len(self.reward_history) > 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = max(np.std(self.reward_history), 1e-6)

    def update_policy(self, gamma=0.99, episode=None, last_effort=None):
        if len(self.rewards) == 0:
            return
        
        # 移除奖励塑造，使用原始奖励
        # if last_effort is not None and episode is not None:
        #     raw_reward = self.rewards[-1]
        #     shaped_reward = self._shape_reward(raw_reward, last_effort, episode)
        #     self.rewards[-1] = shaped_reward
        #     self.recent_shaped_rewards.append(shaped_reward)
            
        # Convert lists to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)  # Remove extra dimensions
        
        # 奖励归一化（如果启用）
        if self.reward_normalization:
            normalized_rewards = [(r - self.reward_mean) / max(self.reward_std, 1.0) for r in self.rewards]
            rewards = torch.tensor(normalized_rewards, dtype=torch.float32)
        else:
            rewards = torch.tensor(self.rewards, dtype=torch.float32)
        
        # 计算GAE优势函数
        if len(self.rewards) > 1:
            # 多步GAE计算
            advantages = []
            gae = 0
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[i + 1]
                delta = rewards[i] + gamma * next_value - values[i]
                gae = delta + gamma * self.gae_lambda * gae
                advantages.insert(0, gae)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = advantages + values.detach()
        else:
            # 单步环境
            returns = rewards
            advantages = returns - values.detach()
        
        # Normalize advantages if we have more than one value
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_div = 0
        total_entropy = 0
        
        # PPO update for multiple epochs
        for epoch in range(self.update_epochs):
            # Forward pass through network
            mean, std, new_values = self.network(states)
            effort_mean = mean * (self.effort_high - self.effort_low) + self.effort_low
            new_values = new_values.squeeze(-1)  # Match dimensions
            
            # Create new distribution
            dist = torch.distributions.Normal(effort_mean, std)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # KL divergence (for monitoring)
            kl_div = torch.exp(old_log_probs - new_log_probs).mean() - 1 - (old_log_probs - new_log_probs).mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_pred_clipped = values + torch.clamp(new_values - values, -self.clip_epsilon, self.clip_epsilon)
            value_loss1 = F.mse_loss(new_values, returns)
            value_loss2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Early stopping based on KL divergence
            if kl_div > 0.02:  # 稍微放松KL约束
                break
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl_div += kl_div.item()
            total_entropy += entropy.item()
        
        # 更新学习率调度器
        if self.scheduler is not None:
            if self.lr_schedule == 'plateau' and last_effort is not None:
                effort_val = last_effort.item() if torch.is_tensor(last_effort) else last_effort
                proximity_score = max(0, 100 - abs(effort_val - self.theoretical_effort))
                self.scheduler.step(proximity_score)
            elif self.lr_schedule in ['cosine_annealing', 'step']:
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
        
        # Log results
        if self.log_path and episode is not None and last_effort is not None:
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                effort_val = last_effort.item() if torch.is_tensor(last_effort) else last_effort
                raw_reward = self.recent_rewards[-1] if self.recent_rewards else 0
                shaped_reward = self.recent_shaped_rewards[-1] if self.recent_shaped_rewards else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.writerow([
                    episode, 
                    round(effort_val, 2), 
                    round(raw_reward.item() if torch.is_tensor(raw_reward) else raw_reward, 2),
                    round(shaped_reward.item() if torch.is_tensor(shaped_reward) else shaped_reward, 2),
                    round(total_policy_loss / self.update_epochs, 4),
                    round(total_value_loss / self.update_epochs, 4),
                    round((total_policy_loss + total_value_loss) / self.update_epochs, 4),
                    round(total_kl_div / self.update_epochs, 6),
                    round(total_entropy / self.update_epochs, 4),
                    f"{current_lr:.6f}",
                    self.curriculum_phase
                ])
        
        # Clear trajectory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
    
    def get_convergence_stats(self):
        """获取收敛统计信息"""
        if len(self.recent_efforts) < 10:
            return None
        
        recent_mean = np.mean(self.recent_efforts[-50:]) if len(self.recent_efforts) >= 50 else np.mean(self.recent_efforts)
        recent_std = np.std(self.recent_efforts[-50:]) if len(self.recent_efforts) >= 50 else np.std(self.recent_efforts)
        
        return {
            'recent_mean_effort': recent_mean,
            'recent_std_effort': recent_std,
            'recent_mean_reward': np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else np.mean(self.recent_rewards)
        }
