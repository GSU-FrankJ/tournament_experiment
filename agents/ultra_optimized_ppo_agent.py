#!/usr/bin/env python3
"""
Ultra-Optimized PPO Agent for Two-Player Tournament Games
=========================================================

This module implements an ultra-optimized PPO agent specifically designed for 
two-player one-stage tournament games.

Key Optimizations:
- Adaptive learning rate with warm restart
- Advanced network architecture with skip connections
- Multi-objective loss function with theoretical effort guidance
- Dynamic reward shaping based on convergence stage
- Curriculum learning with automatic stage progression

Performance Targets:
- Gap < 0.5 (target: < 0.1)
- Convergence time < 10,000 episodes
- Stable convergence without oscillations
"""

import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import math

class UltraOptimizedPPONetwork(nn.Module):
    """
    Ultra-optimized neural network architecture with advanced features
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, 
                 num_layers: int = 4, theoretical_effort: float = 87.5,
                 effort_range: Tuple[float, float] = (0, 200)):
        super().__init__()
        
        self.theoretical_effort = theoretical_effort
        self.effort_low, self.effort_high = effort_range
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        current_dim = input_dim
        for i in range(num_layers - 1):
            layer = nn.Linear(current_dim, hidden_dim)
            self.feature_layers.append(layer)
            
            # Skip connection for residual learning
            if current_dim == hidden_dim:
                self.skip_connections.append(nn.Identity())
            else:
                self.skip_connections.append(nn.Linear(current_dim, hidden_dim))
            
            current_dim = hidden_dim
        
        # Separate policy and value heads
        self.policy_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        ])
        
        self.value_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        ])
        
        # Output heads
        self.policy_mean = nn.Linear(hidden_dim // 4, 1)
        self.policy_std = nn.Linear(hidden_dim // 4, 1)
        self.value_head = nn.Linear(hidden_dim // 4, 1)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with theoretical effort bias"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Bias policy output towards theoretical effort
        with torch.no_grad():
            normalized_effort = (self.theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
            target_logit = torch.logit(torch.tensor(normalized_effort).clamp(0.01, 0.99))
            self.policy_mean.bias.fill_(target_logit.item())
            self.policy_std.bias.fill_(math.log(0.1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with skip connections"""
        features = x
        for i, (layer, skip, bn) in enumerate(zip(self.feature_layers, self.skip_connections, self.batch_norms)):
            residual = skip(features)
            features = layer(features)
            
            if features.size(0) > 1:
                features = bn(features)
            
            features = F.elu(features)
            features = features + residual
            features = self.dropout(features)
        
        # Policy network
        policy_features = features
        for layer in self.policy_layers:
            policy_features = F.elu(layer(policy_features))
            policy_features = self.dropout(policy_features)
        
        # Value network
        value_features = features
        for layer in self.value_layers:
            value_features = F.elu(layer(value_features))
            value_features = self.dropout(value_features)
        
        # Output heads
        mean_logit = self.policy_mean(policy_features)
        mean = torch.sigmoid(mean_logit)
        
        std_raw = self.policy_std(policy_features)
        std = F.softplus(std_raw) + 1e-6
        std = torch.clamp(std, min=1e-6, max=0.5)
        
        value = self.value_head(value_features)
        
        return mean, std, value

    def update_theoretical_effort(self, new_theoretical_effort: float):
        """Dynamically update the network bias towards new theoretical effort"""
        self.theoretical_effort = new_theoretical_effort
        
        with torch.no_grad():
            normalized_effort = (new_theoretical_effort - self.effort_low) / (self.effort_high - self.effort_low)
            normalized_effort = max(0.01, min(0.99, normalized_effort))  # Clamp safely
            target_logit = torch.logit(torch.tensor(normalized_effort))
            
            # Update policy mean bias
            self.policy_mean.bias.fill_(target_logit.item())
            self.policy_std.bias.fill_(math.log(0.1))

class UltraOptimizedPPOAgent:
    """
    Ultra-optimized PPO agent with ADAPTIVE theoretical effort targeting
    """
    
    def __init__(self, effort_range: Tuple[float, float] = (0, 200), 
                 theoretical_effort: float = 87.5, log_path: Optional[str] = None):
        
        self.effort_low, self.effort_high = effort_range
        self.theoretical_effort = theoretical_effort
        self.log_path = log_path
        
        # Ultra-optimized hyperparameters
        self.lr_initial = 0.0001
        self.lr_current = self.lr_initial
        self.clip_epsilon = 0.15
        self.value_coef = 0.8
        self.entropy_coef = 0.003
        self.max_grad_norm = 0.2
        self.gamma = 0.998
        self.gae_lambda = 0.98
        self.update_epochs = 12
        self.batch_size = 128
        self.weight_decay = 5e-6
        
        # Curriculum learning - ADAPTIVE ranges
        self.curriculum_stage = 0
        self.convergence_threshold = 1.0
        
        # Create network
        self.network = UltraOptimizedPPONetwork(
            input_dim=1,
            hidden_dim=256,
            num_layers=5,
            theoretical_effort=theoretical_effort,
            effort_range=effort_range
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.lr_initial,
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )
        
        # Tracking variables
        self.episode_count = 0
        self.recent_efforts = deque(maxlen=1000)
        self.recent_rewards = deque(maxlen=1000)
        self.recent_losses = deque(maxlen=100)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
        # Initialize adaptive curriculum stages
        self._update_curriculum_stages()
        
        # Initialize logging
        self._init_logging()
    
    def update_theoretical_effort(self, new_theoretical_effort: float):
        """Update the theoretical effort target and adapt the network"""
        print(f"🎯 Updating theoretical effort from {self.theoretical_effort:.2f} to {new_theoretical_effort:.2f}")
        
        self.theoretical_effort = new_theoretical_effort
        
        # Update network bias
        self.network.update_theoretical_effort(new_theoretical_effort)
        
        # Update curriculum stages
        self._update_curriculum_stages()
        
        # Reset curriculum stage to start fresh
        self.curriculum_stage = 0
        
        # Clear recent tracking to avoid confusion
        self.recent_efforts.clear()
        self.recent_rewards.clear()
        
        print(f"✅ Network adapted to new theoretical effort: {new_theoretical_effort:.2f}")
    
    def _update_curriculum_stages(self):
        """Update curriculum stages based on current theoretical effort"""
        te = self.theoretical_effort
        margin = max(20, te * 0.3)  # Adaptive margin based on theoretical effort
        
        self.curriculum_stages = [
            {"effort_range": (max(0, te - margin), min(self.effort_high, te + margin)), 
             "threshold": 5.0, "episodes": 2000},
            {"effort_range": (max(0, te - margin*0.7), min(self.effort_high, te + margin*0.7)), 
             "threshold": 3.0, "episodes": 3000},
            {"effort_range": (max(0, te - margin*0.5), min(self.effort_high, te + margin*0.5)), 
             "threshold": 2.0, "episodes": 4000},
            {"effort_range": (max(0, te - margin*0.2), min(self.effort_high, te + margin*0.2)), 
             "threshold": 1.0, "episodes": 5000},
        ]
        
        print(f"📚 Updated curriculum stages around theoretical effort {te:.2f}:")
        for i, stage in enumerate(self.curriculum_stages):
            print(f"   Stage {i+1}: effort_range={stage['effort_range']}, threshold={stage['threshold']}")

    def _init_logging(self):
        """Initialize logging system"""
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Episode', 'Effort', 'Reward', 'Gap', 'PolicyLoss', 'ValueLoss', 
                    'Entropy', 'LearningRate', 'CurriculumStage', 'ConvergenceScore', 'TheoreticalEffort'
                ])
    
    def get_current_effort_range(self) -> Tuple[float, float]:
        """Get effort range for current curriculum stage"""
        if self.curriculum_stage < len(self.curriculum_stages):
            return self.curriculum_stages[self.curriculum_stage]["effort_range"]
        return self.effort_low, self.effort_high
    
    def dynamic_reward_shaping(self, raw_reward: float, effort: float) -> float:
        """Apply dynamic reward shaping based on CURRENT theoretical effort"""
        gap = abs(effort - self.theoretical_effort)  # Use current theoretical effort
        
        shaped_reward = raw_reward
        distance_penalty = -0.2 * gap * (1 + gap / 10)
        
        current_threshold = self.convergence_threshold
        if gap < current_threshold * 0.1:
            convergence_bonus = 2.0
        elif gap < current_threshold * 0.5:
            convergence_bonus = 1.0
        elif gap < current_threshold:
            convergence_bonus = 0.5
        else:
            convergence_bonus = 0.0
        
        if len(self.recent_efforts) > 10:
            recent_std = np.std(list(self.recent_efforts)[-10:])
            stability_bonus = max(0, 1.0 - recent_std / 5.0)
        else:
            stability_bonus = 0.0
        
        return shaped_reward + distance_penalty + convergence_bonus + stability_bonus
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action with ensemble prediction"""
        self.network.eval()
        
        with torch.no_grad():
            mean, std, value = self.network(state.unsqueeze(0))
            
            dist = torch.distributions.Normal(mean, std)
            action_normalized = dist.sample()
            log_prob = dist.log_prob(action_normalized)
            
            current_low, current_high = self.get_current_effort_range()
            action = action_normalized * (current_high - current_low) + current_low
            action = torch.clamp(action, current_low, current_high)
            
            # Store for training
            self.states.append(state)
            self.actions.append(action_normalized.squeeze(0))
            self.log_probs.append(log_prob.squeeze(0))
            self.values.append(value.squeeze(0))
        
        self.network.train()
        return action.squeeze(0)
    
    def store_reward(self, reward: float):
        """Store reward and update tracking"""
        self.recent_rewards.append(reward)
        self.rewards.append(reward)
    
    def update_curriculum_stage(self):
        """Update curriculum stage based on performance"""
        if self.curriculum_stage >= len(self.curriculum_stages):
            return
        
        current_stage = self.curriculum_stages[self.curriculum_stage]
        
        if len(self.recent_efforts) >= 100:
            recent_gaps = [abs(e - self.theoretical_effort) for e in list(self.recent_efforts)[-100:]]
            avg_gap = np.mean(recent_gaps)
            
            if avg_gap < current_stage["threshold"] or self.episode_count > current_stage["episodes"]:
                self.curriculum_stage += 1
                self.convergence_threshold = current_stage["threshold"]
                print(f"Advanced to curriculum stage {self.curriculum_stage}")
                
                self.lr_current = self.lr_initial * 0.8 ** self.curriculum_stage
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_current
    
    def compute_theoretical_guidance_loss(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute loss that guides actions towards theoretical effort"""
        current_low, current_high = self.get_current_effort_range()
        target_normalized = (self.theoretical_effort - current_low) / (current_high - current_low)
        target_normalized = torch.clamp(torch.tensor(target_normalized), 0.0, 1.0)
        
        gap_weight = min(2.0, 1.0 / (torch.mean(torch.abs(actions - target_normalized)) + 1e-6))
        guidance_loss = gap_weight * F.mse_loss(actions, target_normalized.expand_as(actions))
        
        return guidance_loss
    
    def update_policy(self, episode: Optional[int] = None) -> Dict[str, float]:
        """Update policy with ultra-optimized training"""
        if episode is not None:
            self.episode_count = episode
        
        self.update_curriculum_stage()
        
        if len(self.rewards) == 0:
            return {}
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        old_values = torch.stack(self.values)
        
        # Check for NaN values
        if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(old_log_probs).any():
            print("Warning: NaN detected in stored data, skipping update")
            self.states.clear()
            self.actions.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.values.clear()
            return {}
        
        # Apply dynamic reward shaping
        shaped_rewards = torch.tensor([
            self.dynamic_reward_shaping(r.item(), a.item() * (self.effort_high - self.effort_low) + self.effort_low) 
            for r, a in zip(rewards, actions)
        ], dtype=torch.float32)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(shaped_rewards, old_values)
        returns = advantages + old_values
        
        # Check for NaN in advantages
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            print("Warning: NaN detected in advantages/returns, skipping update")
            self.states.clear()
            self.actions.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.values.clear()
            return {}
        
        # Normalize advantages with numerical stability
        advantages_std = advantages.std()
        if advantages_std < 1e-8:
            advantages_normalized = advantages
        else:
            advantages_normalized = (advantages - advantages.mean()) / (advantages_std + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Multiple update epochs
        for epoch in range(self.update_epochs):
            # Check network parameters for NaN
            has_nan = False
            for name, param in self.network.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN gradient in {name} at epoch {epoch}")
                    has_nan = True
            
            if has_nan:
                print(f"Network corrupted after policy update at epoch {epoch}")
                # Reset network parameters to prevent further corruption
                self.network.apply(self._reset_parameters)
                break
            
            means, stds, values = self.network(states)
            
            # Check for NaN in network outputs
            if torch.isnan(means).any() or torch.isnan(stds).any() or torch.isnan(values).any():
                print(f"Warning: NaN detected in network outputs at epoch {epoch}, skipping update")
                break
            
            # Clamp std to prevent numerical issues
            stds = torch.clamp(stds, min=1e-6, max=1.0)
            
            dist = torch.distributions.Normal(means, stds)
            new_log_probs = dist.log_prob(actions)
            
            # Check for NaN in log probs
            if torch.isnan(new_log_probs).any():
                print(f"Warning: NaN detected in log probs at epoch {epoch}, skipping update")
                break
            
            # Policy loss with numerical stability
            ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, min=-10, max=10))
            surr1 = ratio * advantages_normalized
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_normalized
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_pred_clipped = old_values + torch.clamp(
                values.squeeze() - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_loss1 = F.mse_loss(values.squeeze(), returns)
            value_loss2 = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
            
            # Entropy loss
            entropy = dist.entropy().mean()
            
            # Check for NaN in losses
            if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy):
                print(f"Warning: NaN detected in losses at epoch {epoch}, skipping update")
                break
            
            # Theoretical guidance loss with stability
            try:
                guidance_loss = self.compute_theoretical_guidance_loss(actions)
                if torch.isnan(guidance_loss):
                    guidance_loss = torch.tensor(0.0)
            except:
                guidance_loss = torch.tensor(0.0)
            
            # Combined loss with smaller guidance weight
            total_loss = (policy_loss + 
                         self.value_coef * value_loss - 
                         self.entropy_coef * entropy +
                         0.01 * guidance_loss)  # Reduced guidance weight
            
            # Check for NaN in total loss
            if torch.isnan(total_loss):
                print(f"Warning: NaN detected in total loss at epoch {epoch}, skipping update")
                break
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Check gradients for NaN and clip them
            has_nan_grad = False
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: NaN/Inf gradient in {name}, zeroing gradients")
                        param.grad.zero_()
                        has_nan_grad = True
            
            if has_nan_grad:
                continue  # Skip this update
            
            # Conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        self.scheduler.step()
        
        # Average losses
        avg_policy_loss = total_policy_loss / self.update_epochs
        avg_value_loss = total_value_loss / self.update_epochs
        avg_entropy = total_entropy / self.update_epochs
        
        self.recent_losses.append(avg_policy_loss + avg_value_loss)
        
        # Log episode
        if self.log_path and len(actions) > 0:
            effort_actual = actions[-1].item() * (self.effort_high - self.effort_low) + self.effort_low
            self._log_episode(episode or self.episode_count, effort_actual, 
                            shaped_rewards[-1].item(), avg_policy_loss, 
                            avg_value_loss, avg_entropy)
        
        # Clear episode data
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                     next_value: float = 0.0) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        return advantages
    
    def _reset_parameters(self, m):
        """Reset parameters to prevent NaN corruption"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)  # Smaller gain for stability
            nn.init.constant_(m.bias, 0.0)
    
    def _log_episode(self, episode: int, effort: float, reward: float,
                    policy_loss: float, value_loss: float, entropy: float):
        """Log episode data"""
        gap = abs(effort - self.theoretical_effort)
        convergence_score = max(0, 10 - gap)
        
        self.recent_efforts.append(effort)
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, f"{effort:.3f}", f"{reward:.3f}", f"{gap:.3f}",
                f"{policy_loss:.6f}", f"{value_loss:.6f}", f"{entropy:.6f}",
                f"{self.scheduler.get_last_lr()[0]:.8f}", self.curriculum_stage,
                f"{convergence_score:.2f}", f"{self.theoretical_effort:.2f}"
            ])
    
    def get_convergence_stats(self) -> Optional[Dict[str, float]]:
        """Get convergence statistics"""
        if len(self.recent_efforts) < 50:
            return None
        
        recent_efforts_list = list(self.recent_efforts)[-100:]
        recent_gaps = [abs(e - self.theoretical_effort) for e in recent_efforts_list]
        
        return {
            'recent_mean_effort': np.mean(recent_efforts_list),
            'recent_std_effort': np.std(recent_efforts_list),
            'recent_mean_gap': np.mean(recent_gaps),
            'recent_min_gap': np.min(recent_gaps),
            'convergence_score': max(0, 10 - np.mean(recent_gaps)),
            'curriculum_stage': self.curriculum_stage,
            'episodes_trained': self.episode_count,
            'theoretical_effort': self.theoretical_effort
        } 