import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple

class OptimizedPPONetwork(nn.Module):
    """
    Configurable PPO network with support for different architectures and techniques
    """
    
    def __init__(self, config: Dict, input_dim=1):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.activation = config.get('activation', 'relu')
        self.use_layer_norm = config.get('use_layer_norm', True)
        self.use_residual = config.get('use_residual', False)
        self.separate_networks = config.get('separate_networks', True)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'tanh':
            self.activation_fn = torch.tanh
        elif self.activation == 'elu':
            self.activation_fn = F.elu
        else:
            self.activation_fn = F.relu
        
        # Build networks
        self._build_networks(input_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _build_networks(self, input_dim):
        """Build policy and value networks based on configuration"""
        
        if self.separate_networks:
            # Separate policy and value networks
            self.policy_layers = self._build_mlp(input_dim, self.hidden_dim, self.num_layers)
            self.policy_mean = nn.Linear(self.hidden_dim, 1)
            self.log_std = nn.Parameter(torch.log(torch.ones(1) * 0.5))
            
            self.value_layers = self._build_mlp(input_dim, self.hidden_dim, self.num_layers)
            self.value_head = nn.Linear(self.hidden_dim, 1)
        else:
            # Shared network with separate heads
            self.shared_layers = self._build_mlp(input_dim, self.hidden_dim, self.num_layers)
            self.policy_mean = nn.Linear(self.hidden_dim, 1)
            self.log_std = nn.Parameter(torch.log(torch.ones(1) * 0.5))
            self.value_head = nn.Linear(self.hidden_dim, 1)
        
        # Dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        # Layer normalization
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
            ])
    
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        """Build MLP layers"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        return nn.ModuleList(layers)
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialization
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Special initialization for policy output to bias towards theoretical value
        if hasattr(self, 'policy_mean'):
            with torch.no_grad():
                # Initialize to output around 0.875 (87.5/100)
                target_logit = torch.logit(torch.tensor(0.875))
                self.policy_mean.bias.fill_(target_logit.item())
                self.policy_mean.weight.fill_(0.01)
    
    def _forward_mlp(self, x, layers, use_residual=False):
        """Forward pass through MLP with optional residual connections"""
        for i, layer in enumerate(layers):
            if use_residual and i > 0 and x.shape[-1] == layer.weight.shape[1]:
                # Residual connection
                residual = x
                x = layer(x)
                if self.use_layer_norm and i < len(self.layer_norms):
                    x = self.layer_norms[i](x)
                x = self.activation_fn(x)
                if hasattr(self, 'dropout'):
                    x = self.dropout(x)
                x = x + residual
            else:
                x = layer(x)
                if self.use_layer_norm and i < len(self.layer_norms):
                    x = self.layer_norms[i](x)
                x = self.activation_fn(x)
                if hasattr(self, 'dropout'):
                    x = self.dropout(x)
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        if self.separate_networks:
            # Policy network
            policy_x = self._forward_mlp(x, self.policy_layers, self.use_residual)
            mean = torch.sigmoid(self.policy_mean(policy_x))
            std = torch.exp(self.log_std).expand_as(mean)
            
            # Value network
            value_x = self._forward_mlp(x, self.value_layers, self.use_residual)
            value = self.value_head(value_x)
        else:
            # Shared network
            shared_x = self._forward_mlp(x, self.shared_layers, self.use_residual)
            mean = torch.sigmoid(self.policy_mean(shared_x))
            std = torch.exp(self.log_std).expand_as(mean)
            value = self.value_head(shared_x)
        
        return mean, std, value

class OptimizedPPOAgent:
    """
    Optimized PPO agent with configurable hyperparameters and advanced techniques
    """
    
    def __init__(self, config: Dict, effort_range=(0, 100), theoretical_effort=87.5, log_path=None):
        self.config = config
        self.effort_low, self.effort_high = effort_range
        self.theoretical_effort = theoretical_effort
        self.log_path = log_path
        
        # Extract hyperparameters from config
        self.lr = config.get('learning_rate', 1e-4)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.update_epochs = config.get('update_epochs', 8)
        self.batch_size = config.get('batch_size', 128)
        self.weight_decay = config.get('weight_decay', 1e-5)
        
        # Advanced features
        self.reward_normalization = config.get('reward_normalization', True)
        self.observation_normalization = config.get('observation_normalization', False)
        self.lr_schedule = config.get('lr_schedule', 'reduce_on_plateau')
        self.lr_decay_factor = config.get('lr_decay_factor', 0.9)
        self.lr_patience = config.get('lr_patience', 1500)
        
        # Create network
        self.network = OptimizedPPONetwork(config)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.lr, 
            eps=1e-5, 
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        self._setup_scheduler()
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []
        
        # Tracking variables
        self.recent_efforts = []
        self.recent_rewards = []
        self.episode_count = 0
        
        # Normalization statistics
        if self.reward_normalization:
            self.reward_mean = 0.0
            self.reward_std = 1.0
            self.reward_history = []
        
        if self.observation_normalization:
            self.obs_mean = 0.0
            self.obs_std = 1.0
            self.obs_history = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.lr_schedule == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=self.lr_decay_factor, 
                patience=self.lr_patience, verbose=False
            )
        elif self.lr_schedule == 'linear_decay':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000
            )
        elif self.lr_schedule == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10000, eta_min=self.lr * 0.1
            )
        else:
            self.scheduler = None
    
    def _setup_logging(self):
        """Setup CSV logging"""
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Episode", "Effort", "Raw_Reward", "Normalized_Reward", 
                    "Policy_Loss", "Value_Loss", "Total_Loss", "KL_Div", 
                    "Entropy", "LR", "Advantage_Mean", "Advantage_Std"
                ])
    
    def normalize_observation(self, obs):
        """Normalize observation if enabled"""
        if not self.observation_normalization:
            return obs
        
        obs_val = obs.item() if torch.is_tensor(obs) else obs
        self.obs_history.append(obs_val)
        
        if len(self.obs_history) > 1000:
            self.obs_history.pop(0)
        
        if len(self.obs_history) > 10:
            self.obs_mean = np.mean(self.obs_history)
            self.obs_std = max(np.std(self.obs_history), 1e-6)
            
            normalized = (obs_val - self.obs_mean) / self.obs_std
            return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        
        return obs
    
    def normalize_reward(self, reward):
        """Normalize reward if enabled"""
        if not self.reward_normalization:
            return reward
        
        reward_val = reward.item() if torch.is_tensor(reward) else reward
        self.reward_history.append(reward_val)
        
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        if len(self.reward_history) > 10:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = max(np.std(self.reward_history), 1e-6)
            
            normalized = (reward_val - self.reward_mean) / self.reward_std
            return normalized
        
        return reward_val
    
    def select_action(self, state):
        """Select action using current policy"""
        # Normalize observation if enabled
        state = self.normalize_observation(state)
        
        with torch.no_grad():
            mean, std, value = self.network(state)
            
            # Scale to effort range
            effort_mean = mean * (self.effort_high - self.effort_low) + self.effort_low
            
            # Create distribution and sample
            dist = torch.distributions.Normal(effort_mean, std)
            action = dist.sample()
            action = torch.clamp(action, self.effort_low, self.effort_high)
            log_prob = dist.log_prob(action)
            
            # Store trajectory data
            self.states.append(state.detach().clone())
            self.actions.append(action.detach().clone())
            self.log_probs.append(log_prob.detach().clone())
            self.values.append(value.detach().clone())
            
            return action
    
    def store_reward(self, reward):
        """Store reward for trajectory"""
        # Normalize reward if enabled
        normalized_reward = self.normalize_reward(reward)
        self.rewards.append(normalized_reward)
        
        # Track recent performance
        effort_val = self.actions[-1].item() if self.actions else 0
        self.recent_efforts.append(effort_val)
        self.recent_rewards.append(normalized_reward)
        
        # Keep only recent history
        if len(self.recent_efforts) > 1000:
            self.recent_efforts.pop(0)
            self.recent_rewards.pop(0)
    
    def compute_gae(self, rewards, values, next_value=0):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Convert to tensors if needed
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        if not torch.is_tensor(values):
            values = torch.stack(values).squeeze()
        
        # Add next value for bootstrapping
        values_with_next = torch.cat([values, torch.tensor([next_value])])
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] - values_with_next[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        
        return advantages, returns
    
    def update_policy(self, episode=None, last_effort=None):
        """Update policy using PPO algorithm"""
        if len(self.rewards) == 0:
            return
        
        self.episode_count += 1
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(self.rewards, self.values)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values).squeeze()
        
        # Store for logging
        self.advantages = advantages
        self.returns = returns
        
        # Multiple epochs of updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        
        for epoch in range(self.update_epochs):
            # Forward pass
            means, stds, values = self.network(states)
            
            # Scale means to effort range
            effort_means = means * (self.effort_high - self.effort_low) + self.effort_low
            
            # Create distribution
            dist = torch.distributions.Normal(effort_means, stds)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute policy loss (PPO clipped objective)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Compute total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Accumulate losses for logging
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            
            # Compute KL divergence for monitoring
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean()
                total_kl_div += kl_div.item()
        
        # Update learning rate scheduler
        if self.scheduler:
            if self.lr_schedule == 'reduce_on_plateau':
                # Use negative gap as metric (higher is better)
                current_performance = -abs(last_effort.item() - self.theoretical_effort) if last_effort else 0
                self.scheduler.step(current_performance)
            else:
                self.scheduler.step()
        
        # Log results
        self._log_episode(episode, last_effort, total_policy_loss, total_value_loss, 
                         total_entropy, total_kl_div)
        
        # Clear trajectory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
    
    def _log_episode(self, episode, last_effort, policy_loss, value_loss, entropy, kl_div):
        """Log episode results"""
        if not self.log_path:
            return
        
        effort_val = last_effort.item() if last_effort is not None else 0
        raw_reward = self.recent_rewards[-1] if self.recent_rewards else 0
        normalized_reward = raw_reward
        
        current_lr = self.optimizer.param_groups[0]['lr']
        advantage_mean = self.advantages.mean().item() if len(self.advantages) > 0 else 0
        advantage_std = self.advantages.std().item() if len(self.advantages) > 0 else 0
        
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, effort_val, raw_reward, normalized_reward,
                policy_loss / self.update_epochs, value_loss / self.update_epochs,
                (policy_loss + value_loss) / self.update_epochs, kl_div / self.update_epochs,
                entropy / self.update_epochs, current_lr, advantage_mean, advantage_std
            ])
    
    def get_convergence_stats(self):
        """Get convergence statistics"""
        if len(self.recent_efforts) < 100:
            return None
        
        recent_window = min(500, len(self.recent_efforts))
        recent_efforts = self.recent_efforts[-recent_window:]
        recent_rewards = self.recent_rewards[-recent_window:]
        
        return {
            'recent_mean_effort': np.mean(recent_efforts),
            'recent_std_effort': np.std(recent_efforts),
            'recent_mean_reward': np.mean(recent_rewards),
            'recent_std_reward': np.std(recent_rewards),
            'episodes_trained': self.episode_count,
            'current_lr': self.optimizer.param_groups[0]['lr']
        } 