import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):  # 增大网络容量
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, 1)  # output effort mean
        self.log_std = nn.Parameter(torch.log(torch.ones(1) * 10.0))  # 可学习的标准差
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 更好的权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                
        # 特殊初始化最后一层，使其输出接近0.875（对应87.5%的理论值）
        with torch.no_grad():
            # 设置最后一层的bias，使sigmoid输出约为0.875
            target_logit = torch.logit(torch.tensor(0.875))  # 理论值/100的logit
            self.fc4.bias.fill_(target_logit.item())
            self.fc4.weight.fill_(0.01)  # 小权重

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        mean = torch.sigmoid(self.fc4(x))  # 输出0-1之间的值
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

class REINFORCEAgent:
    def __init__(self, lr=1e-4, effort_range=(0, 100), log_path=None, baseline_decay=0.95, theoretical_effort=87.5):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.8, patience=2000, verbose=True)
        
        self.effort_low, self.effort_high = effort_range
        self.saved_log_probs = []
        self.rewards = []
        self.log_path = log_path
        self.theoretical_effort = theoretical_effort
        
        # Baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        
        # 收敛监控
        self.recent_efforts = []
        self.recent_rewards = []
        self.recent_shaped_rewards = []
        
        # 课程学习参数
        self.curriculum_phase = 0
        self.phase_transitions = [10000, 25000]  # 阶段转换点
        
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Effort", "Raw_Reward", "Shaped_Reward", "Loss", "Baseline", "LR", "Curriculum_Phase"])

    def _shape_reward(self, raw_reward, effort_value, episode):
        """奖励塑造：引导向理论最优值"""
        effort_val = effort_value.item() if torch.is_tensor(effort_value) else effort_value
        
        # 温和的距离奖励 - 鼓励接近理论值但不要太强
        distance_to_optimal = abs(effort_val - self.theoretical_effort)
        distance_reward = -distance_to_optimal * 0.02  # 降低强度
        
        # 更温和的课程学习奖励
        if self.curriculum_phase == 0:
            # 第一阶段：鼓励探索中等努力水平
            target_range = (50, 90)
            if target_range[0] <= effort_val <= target_range[1]:
                curriculum_reward = 1.0  # 降低奖励强度
            else:
                curriculum_reward = -abs(effort_val - 70) * 0.01
        elif self.curriculum_phase == 1:
            # 第二阶段：鼓励更接近理论值
            target_range = (70, 95)
            if target_range[0] <= effort_val <= target_range[1]:
                curriculum_reward = 2.0  # 降低奖励强度
            else:
                curriculum_reward = -abs(effort_val - self.theoretical_effort) * 0.02
        else:
            # 最终阶段：鼓励精确值，但不要太强
            if abs(effort_val - self.theoretical_effort) < 5:
                curriculum_reward = 3.0  # 大幅降低奖励强度
            else:
                curriculum_reward = -abs(effort_val - self.theoretical_effort) * 0.05
        
        # 结合所有奖励，确保原始奖励仍然是主要部分
        shaped_reward = raw_reward + 0.3 * (distance_reward + curriculum_reward)  # 降低塑造权重
        
        # 根据episode更新课程阶段
        if episode in self.phase_transitions:
            self.curriculum_phase += 1
            print(f"Curriculum learning phase advanced to {self.curriculum_phase}")
        
        return shaped_reward

    def select_action(self, state):
        mean, std = self.policy(state)
        effort_mean = mean * (self.effort_high - self.effort_low) + self.effort_low
        
        # 在早期添加一些引导的噪声
        if len(self.recent_efforts) < 1000:
            # 早期阶段，添加朝向理论值的偏置
            bias_factor = 0.1 * (self.theoretical_effort - effort_mean) / self.effort_high
            effort_mean = effort_mean + bias_factor * torch.randn_like(effort_mean) * 10
        
        # 使用正态分布采样
        dist = torch.distributions.Normal(effort_mean, std)
        action = dist.sample()
        action = torch.clamp(action, self.effort_low, self.effort_high)
        log_prob = dist.log_prob(action)
        
        self.saved_log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self, gamma=0.99, episode=None, last_effort=None):
        if not self.rewards:
            return
        
        # 应用奖励塑造
        if last_effort is not None and episode is not None:
            raw_reward = self.rewards[-1]
            shaped_reward = self._shape_reward(raw_reward, last_effort, episode)
            self.rewards[-1] = shaped_reward
            self.recent_shaped_rewards.append(shaped_reward)
            
        # 计算returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 更新baseline (移动平均)
        current_return = returns[0].item()
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * current_return
        
        # 使用baseline减少方差
        advantages = returns - self.baseline
        
        # 标准化advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算policy loss
        loss = []
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            loss.append(-log_prob * advantage)
        
        if loss:
            self.optimizer.zero_grad()
            total_loss = torch.stack(loss).sum()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新学习率调度器（基于接近理论值的程度）
            if last_effort is not None:
                effort_val = last_effort.item() if torch.is_tensor(last_effort) else last_effort
                proximity_score = max(0, 100 - abs(effort_val - self.theoretical_effort))
                self.scheduler.step(proximity_score)
            
            # 监控收敛
            if last_effort is not None:
                effort_val = last_effort.item() if torch.is_tensor(last_effort) else last_effort
                reward_val = self.rewards[-1].item() if torch.is_tensor(self.rewards[-1]) else self.rewards[-1]
                
                self.recent_efforts.append(effort_val)
                self.recent_rewards.append(reward_val)
                
                # 保持最近100个结果
                if len(self.recent_efforts) > 100:
                    self.recent_efforts.pop(0)
                    self.recent_rewards.pop(0)

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
                        round(total_loss.item(), 4),
                        round(self.baseline, 4),
                        f"{current_lr:.6f}",
                        self.curriculum_phase
                    ])

        self.rewards.clear()
        self.saved_log_probs.clear()
    
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