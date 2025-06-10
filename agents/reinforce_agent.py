import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # output effort value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # effort 范围归一化
        return x

class REINFORCEAgent:
    def __init__(self, lr=1e-3, effort_range=(0, 100), log_path=None):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.effort_low, self.effort_high = effort_range
        self.saved_log_probs = []
        self.rewards = []
        self.log_path = log_path

        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Effort", "Reward", "Loss"])  # header

    #     self._initialize_to_midpoint()

    # def _initialize_to_midpoint(self):
    #     with torch.no_grad():
    #         midpoint = 0.5 
    #         hidden_out = torch.zeros_like(self.policy.fc1.bias)
    #         self.policy.fc1.weight.fill_(0.0)
    #         self.policy.fc1.bias.copy_(hidden_out)
    #         self.policy.fc2.weight.fill_(0.0)
    #         self.policy.fc2.bias.fill_(torch.logit(torch.tensor(midpoint)))

    def select_action(self, state):
        effort_prob = self.policy(state)
        effort = effort_prob * (self.effort_high - self.effort_low) + self.effort_low
        m = torch.distributions.Normal(effort, 1.0)  
        action = m.sample()
        action = torch.clamp(action, self.effort_low, self.effort_high)
        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self, gamma=1.0, episode=None, last_effort=None):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            loss.append(-log_prob * R)
        if loss:
            self.optimizer.zero_grad()
            total_loss = torch.stack(loss).sum()
            total_loss.backward()
            self.optimizer.step()

            if self.log_path and episode is not None and last_effort is not None:
                with open(self.log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, round(last_effort.item(), 2), round(self.rewards[-1].item(), 2), round(total_loss.item(), 4)])

        self.rewards.clear()
        self.saved_log_probs.clear()