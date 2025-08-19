#!/usr/bin/env python3
"""
Three Players Self-Play PPO Agent
=================================

This module implements a PPO agent specifically designed for three players self-play scenarios.
Each agent can independently learn optimal strategies without being constrained by symmetric equilibrium assumptions.

Key Features:
- Independent learning for each player
- No assumption of symmetric equilibrium
- Adaptive learning based on opponent strategies
- Support for asymmetric equilibrium discovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import os
import csv

class SelfPlayPolicyNetwork(nn.Module):
    """
    Policy network specifically designed for self-play scenarios.
    
    This network learns to adapt to different opponent strategies and
    can discover asymmetric equilibria.
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 256, 
                 num_layers: int = 4, activation: str = 'relu',
                 dropout_rate: float = 0.1, action_dim: int = 1):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, hidden_dim // 2))
        
        self.layers = nn.ModuleList(layers)
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.std_head = nn.Linear(hidden_dim // 2, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Initialize std head to produce reasonable std values
        nn.init.constant_(self.std_head.bias, 0.5)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                x = self.activation(x)
        
        # Use tanh for mean to map to [-1, 1] then scale to [0, 1]
        mean = (torch.tanh(self.mean_head(x)) + 1.0) / 2.0
        # Use softplus for std to ensure it's always positive
        std = F.softplus(self.std_head(x)) + 1e-6
        std = torch.clamp(std, min=1e-6, max=2.0)
        
        return mean, std

class SelfPlayValueNetwork(nn.Module):
    """
    Value network for self-play scenarios.
    
    This network estimates the expected return for the current state,
    helping the agent understand the value of different strategies.
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 256,
                 num_layers: int = 4, activation: str = 'relu',
                 dropout_rate: float = 0.1):
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
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, hidden_dim // 2))
        layers.append(nn.Linear(hidden_dim // 2, 1))
        
        self.layers = nn.ModuleList(layers)
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
            if isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x

class SelfPlayPPOAgent:
    """
    PPO agent designed for self-play in three players scenarios.
    
    This agent can learn optimal strategies independently and adapt
    to different opponent behaviors without assuming symmetry.
    """
    
    def __init__(self, player_id: int, effort_range: Tuple[float, float] = (0, 200),
                 learning_rate: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
                  max_grad_norm: float = 0.5, log_path: Optional[str] = None,
                  initial_offset: float = 0.0):
        """
        Initialize the self-play PPO agent.
        
        Args:
            player_id: ID of this player (0, 1, or 2)
            effort_range: Range of valid effort values
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            log_path: Path for logging training progress
        """
        self.player_id = player_id
        self.effort_range = effort_range
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # Small per-agent offset in normalized action space to ensure non-identical initial efforts
        # This avoids symmetric initialization among agents and encourages asymmetric exploration.
        self.initial_offset = float(initial_offset)
        
        # Network setup
        self.policy_net = SelfPlayPolicyNetwork(input_dim=1, hidden_dim=256, num_layers=4)
        self.value_net = SelfPlayValueNetwork(input_dim=1, hidden_dim=256, num_layers=4)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.recent_efforts = deque(maxlen=1000)
        self.recent_rewards = deque(maxlen=1000)
        self.recent_utilities = deque(maxlen=1000)
        
        # Logging
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'effort', 'reward', 'utility', 'policy_loss', 'value_loss'])
        
        print(f"🤖 SelfPlayPPOAgent {player_id} initialized:")
        print(f"   - Effort range: {effort_range}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Log path: {log_path}")
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action (effort level) based on the current state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Selected effort level as tensor
        """
        # Get action distribution over normalized effort in [0, 1]
        mean, std = self.policy_net(state)
        
        # Create normal distribution in normalized space
        dist = torch.distributions.Normal(mean, std)
        
        # Sample normalized action, apply per-agent offset, and clamp to [0, 1]
        normalized_action = dist.sample()
        normalized_action = torch.clamp(normalized_action + self.initial_offset, 0.0, 1.0)
        
        # Scale to actual effort range
        low, high = self.effort_range
        action = normalized_action * (high - low) + low
        action = torch.clamp(action, low, high)
        
        # Store for training (store actual effort for logging, training re-normalizes)
        self.recent_efforts.append(action.item())
        
        return action
    
    def store_experience(self, effort: float, reward: float, utility: float):
        """
        Store experience for training.
        
        Args:
            effort: Effort level chosen
            reward: Reward received
            utility: Utility received
        """
        self.recent_rewards.append(reward)
        self.recent_utilities.append(utility)
    
    def update_policy(self, episode: int = None):
        """
        Update the policy using stored experiences.
        
        Args:
            episode: Current episode number
        """
        if len(self.recent_rewards) < 50:
            return  # Need more data
        
        # Prepare training data
        states = torch.tensor([[0.0]] * len(self.recent_rewards), dtype=torch.float32)
        efforts = torch.tensor(list(self.recent_efforts)[-len(self.recent_rewards):], dtype=torch.float32)
        rewards = torch.tensor(list(self.recent_rewards), dtype=torch.float32)
        
        # Normalize efforts to [0, 1] for training
        low, high = self.effort_range
        normalized_efforts = (efforts - low) / (high - low)
        normalized_efforts = torch.clamp(normalized_efforts, 0.0, 1.0)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            # Policy loss
            mean, std = self.policy_net(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(normalized_efforts)
            
            # Value loss
            values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(values, rewards)
            
            # Policy loss with clipping
            ratio = torch.exp(log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        # Log progress
        if self.log_path and episode is not None:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, efforts[-1].item(), rewards[-1].item(), 
                               list(self.recent_utilities)[-1], policy_loss.item(), value_loss.item()])
        
        self.episode_count += 1
    
    def _compute_gae(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards
            
        Returns:
            Tensor of advantages
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = rewards[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - rewards[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def get_convergence_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get convergence statistics for this agent.
        
        Returns:
            Dictionary containing convergence statistics
        """
        if len(self.recent_efforts) < 100:
            return None
        
        recent_efforts = list(self.recent_efforts)[-100:]
        recent_rewards = list(self.recent_rewards)[-100:]
        
        # Calculate statistics
        mean_effort = np.mean(recent_efforts)
        std_effort = np.std(recent_efforts)
        mean_reward = np.mean(recent_rewards)
        
        # Check for convergence (effort stability)
        effort_cv = std_effort / (mean_effort + 1e-8)
        converged = effort_cv < 0.1  # Coefficient of variation threshold
        
        stats = {
            "player_id": self.player_id,
            "mean_effort": float(mean_effort),
            "std_effort": float(std_effort),
            "mean_reward": float(mean_reward),
            "effort_cv": float(effort_cv),
            "converged": converged,
            "episode_count": self.episode_count
        }
        
        return stats
    
    def get_recent_effort(self) -> float:
        """
        Get the most recent effort level.
        
        Returns:
            Most recent effort level
        """
        if len(self.recent_efforts) > 0:
            return list(self.recent_efforts)[-1]
        else:
            return (self.effort_range[0] + self.effort_range[1]) / 2
    
    def reset_learning_state(self):
        """Reset the learning state for new training session."""
        self.recent_efforts.clear()
        self.recent_rewards.clear()
        self.recent_utilities.clear()
        self.episode_count = 0
    
    def update_parameters(self, q_value: float, effort_range: tuple, theoretical_effort: float):
        """
        Dynamically adjust agent configuration based on new parameters.
        Mirrors adaptive hooks used in enhanced PPO agents.
        
        - Update theoretical target used only for initialization biasing
        - Adjust action scaling to new effort range
        - Reset learning buffers to avoid stale policy bias
        """
        self.effort_range = effort_range
        # Light adaptive initialization: nudge policy mean towards normalized theoretical effort
        low, high = effort_range
        if high > low:
            normalized_theoretical = float(np.clip((theoretical_effort - low) / (high - low), 0.0, 1.0))
            # Apply a small bias to the last layer bias to point near theoretical effort
            with torch.no_grad():
                # Move mean head bias slightly towards desired normalized mean
                current_bias = self.policy_net.mean_head.bias.data.clone()
                target = torch.full_like(current_bias, normalized_theoretical)
                self.policy_net.mean_head.bias.data = 0.9 * current_bias + 0.1 * target
        # Optionally adjust exploration scale mildly via std_head bias
        with torch.no_grad():
            self.policy_net.std_head.bias.data = torch.clamp(self.policy_net.std_head.bias.data, min=0.1, max=1.0)
        # Reset buffers to adapt quickly to new setting
        self.reset_learning_state()
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'player_id': self.player_id,
            'effort_range': self.effort_range
        }, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict']) 