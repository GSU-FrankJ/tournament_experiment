import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

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

class EnhancedPPOAgent:
    """Enhanced PPO agent supporting both discrete and continuous action spaces"""
    
    def __init__(self, action_space: ActionSpace, input_dim: int = 1,
                 lr: float = 3e-4, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, gamma: float = 0.99,
                 gae_lambda: float = 0.95, update_epochs: int = 4,
                 batch_size: int = 64, hidden_dim: int = 128,
                 num_layers: int = 3, activation: str = 'relu',
                 dropout_rate: float = 0.05, weight_decay: float = 1e-5,
                 lr_schedule: str = 'constant', separate_networks: bool = True,
                 reward_normalization: bool = True, log_path: Optional[str] = None):
        
        self.action_space = action_space
        self.input_dim = input_dim
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.separate_networks = separate_networks
        self.reward_normalization = reward_normalization
        self.log_path = log_path
        
        # Create networks based on action space type
        if action_space.action_type == 'continuous':
            self.policy_network = ContinuousPolicyNetwork(
                input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                activation=activation, dropout_rate=dropout_rate,
                action_dim=action_space.shape[0]
            )
        elif action_space.action_type == 'discrete':
            self.policy_network = DiscretePolicyNetwork(
                input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                activation=activation, dropout_rate=dropout_rate,
                num_actions=action_space.n
            )
        else:
            raise ValueError(f"Unsupported action space type: {action_space.action_type}")
        
        # Value network (always separate for better modularity)
        self.value_network = ValueNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            activation=activation, dropout_rate=dropout_rate
        )
        
        # Optimizers
        if separate_networks:
            self.policy_optimizer = optim.Adam(
                self.policy_network.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.value_optimizer = optim.Adam(
                self.value_network.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            # Combined optimizer for shared parameters
            all_params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            self.optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self._setup_scheduler(lr_schedule)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = []
        
        # Convergence monitoring
        self.recent_efforts = []
        self.recent_rewards = []
        
        # Initialize logging
        if self.log_path:
            self._init_logging()
    
    def _setup_scheduler(self, lr_schedule: str):
        """Setup learning rate scheduler"""
        if lr_schedule == 'cosine_annealing':
            if self.separate_networks:
                self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.policy_optimizer, T_max=1000, eta_min=self.lr * 0.1
                )
                self.value_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.value_optimizer, T_max=1000, eta_min=self.lr * 0.1
                )
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=1000, eta_min=self.lr * 0.1
                )
        elif lr_schedule == 'step':
            if self.separate_networks:
                self.policy_scheduler = optim.lr_scheduler.StepLR(
                    self.policy_optimizer, step_size=500, gamma=0.9
                )
                self.value_scheduler = optim.lr_scheduler.StepLR(
                    self.value_optimizer, step_size=500, gamma=0.9
                )
            else:
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=500, gamma=0.9
                )
        else:
            self.policy_scheduler = None
            self.value_scheduler = None
            self.scheduler = None
    
    def _init_logging(self):
        """Initialize logging"""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "Action", "Reward", "Policy_Loss", "Value_Loss", 
                "Total_Loss", "KL_Div", "Entropy", "LR"
            ])
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action using current policy"""
        with torch.no_grad():
            # Get action distribution
            dist = self.policy_network.get_action_distribution(state)
            
            # Sample action
            if self.action_space.action_type == 'continuous':
                action = dist.sample()
                # Scale to action range
                if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                    action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
                    action = torch.clamp(action, self.action_space.low, self.action_space.high)
                # Ensure correct shape for continuous actions (1,)
                action = action.squeeze(0) if action.dim() > 1 else action
            else:  # discrete
                action = dist.sample()
                # Ensure scalar for discrete actions
                action = action.squeeze() if action.dim() > 0 else action
            
            log_prob = dist.log_prob(action)
            value = self.value_network(state)
            
            # Store trajectory data
            self.states.append(state.detach().clone())
            self.actions.append(action.detach().clone())
            self.log_probs.append(log_prob.detach().clone())
            self.values.append(value.detach().clone())
            
            return action
    
    def store_reward(self, reward: Union[torch.Tensor, float]):
        """Store reward for trajectory"""
        reward_val = reward.item() if torch.is_tensor(reward) else reward
        
        # Normalize reward if enabled
        if self.reward_normalization:
            self.reward_history.append(reward_val)
            if len(self.reward_history) > 1000:
                self.reward_history.pop(0)
            
            # Update statistics immediately when we have enough data
            if len(self.reward_history) >= 2:  # Changed from > 10 to >= 2
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = max(np.std(self.reward_history), 1e-6)
                reward_val = (reward_val - self.reward_mean) / self.reward_std
        
        self.rewards.append(reward_val)
        
        # Track recent performance
        if self.actions:
            if self.action_space.action_type == 'continuous':
                effort_val = self.actions[-1].item()
            else:
                effort_val = self.actions[-1].item()  # For discrete, this is the action index
            
            self.recent_efforts.append(effort_val)
            self.recent_rewards.append(reward_val)
            
            # Keep only recent history
            if len(self.recent_efforts) > 1000:
                self.recent_efforts.pop(0)
                self.recent_rewards.pop(0)
    
    def update_policy(self, episode: Optional[int] = None) -> Dict[str, float]:
        """Update policy using PPO algorithm"""
        if len(self.rewards) == 0:
            return {}
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        old_values = torch.stack(self.values).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        
        # Compute GAE
        advantages, returns = GAECalculator.compute_gae(
            rewards, old_values, self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        
        for epoch in range(self.update_epochs):
            # Check network health before forward pass
            first_layer = self.policy_network.layers[0]
            if torch.isnan(first_layer.weight).any() or torch.isnan(first_layer.bias).any():
                print(f"Network corrupted before epoch {epoch}")
                break
            
            # Forward pass
            dist = self.policy_network.get_action_distribution(states)
            
            # Check for NaN values in distribution parameters
            if self.action_space.action_type == 'continuous':
                mean, std = self.policy_network.forward(states)
                if torch.isnan(mean).any() or torch.isnan(std).any():
                    print(f"Warning: NaN detected in policy network output at epoch {epoch}")
                    print(f"Network weights corrupted: {torch.isnan(first_layer.weight).any()}")
                    break
            
            new_log_probs = dist.log_prob(actions)
            # Ensure log_probs have the correct shape - should be (batch_size,) not (batch_size, batch_size, ...)
            if new_log_probs.dim() > 1:
                # Take diagonal to get the correct log probabilities for each state-action pair
                if new_log_probs.shape[0] == new_log_probs.shape[1]:
                    new_log_probs = torch.diagonal(new_log_probs, dim1=0, dim2=1)
                else:
                    new_log_probs = new_log_probs.squeeze()
            
            entropy = dist.entropy().mean()
            new_values = self.value_network(states).squeeze(-1)
            
            # Check for NaN in computed values
            if torch.isnan(new_log_probs).any() or torch.isnan(entropy) or torch.isnan(new_values).any():
                print(f"Warning: NaN detected in forward pass at epoch {epoch}")
                break
            
            # Compute losses
            policy_loss = PPOLoss.compute_policy_loss(
                new_log_probs, old_log_probs, advantages, self.clip_epsilon
            )
            value_loss = PPOLoss.compute_value_loss(
                new_values, old_values, returns, self.clip_epsilon
            )
            
            # Check for NaN in losses
            if torch.isnan(policy_loss) or torch.isnan(value_loss):
                print(f"Warning: NaN detected in loss computation at epoch {epoch}")
                break
            
            # KL divergence for monitoring
            kl_div = (old_log_probs - new_log_probs).mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Early stopping based on KL divergence
            if kl_div > 0.02:
                break
            
            # Backward pass
            if self.separate_networks:
                # Update policy network
                self.policy_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward(retain_graph=True)
                
                # Check for NaN in gradients
                for name, param in self.policy_network.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient in policy network {name} at epoch {epoch}")
                            break
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_network.parameters(), self.max_grad_norm
                    )
                self.policy_optimizer.step()
                
                # Check network health after policy update
                if torch.isnan(first_layer.weight).any() or torch.isnan(first_layer.bias).any():
                    print(f"Network corrupted after policy update at epoch {epoch}")
                    break
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                
                # Check for NaN in value gradients
                for name, param in self.value_network.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient in value network {name} at epoch {epoch}")
                        break
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.value_network.parameters(), self.max_grad_norm
                    )
                self.value_optimizer.step()
            else:
                # Update combined network
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    all_params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl_div += kl_div.item()
        
        # Update learning rate schedulers
        if self.separate_networks:
            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.value_scheduler:
                self.value_scheduler.step()
        else:
            if self.scheduler:
                self.scheduler.step()
        
        # Log results
        if self.log_path and episode is not None:
            self._log_episode(episode, total_policy_loss, total_value_loss, 
                            total_entropy, total_kl_div)
        
        # Clear trajectory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        
        return {
            'policy_loss': total_policy_loss / self.update_epochs,
            'value_loss': total_value_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs,
            'kl_div': total_kl_div / self.update_epochs
        }
    
    def _log_episode(self, episode: int, policy_loss: float, value_loss: float,
                    entropy: float, kl_div: float):
        """Log episode results"""
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            action_val = self.recent_efforts[-1] if self.recent_efforts else 0
            reward_val = self.recent_rewards[-1] if self.recent_rewards else 0
            
            if self.separate_networks:
                current_lr = self.policy_optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            writer.writerow([
                episode, round(action_val, 2), round(reward_val, 4),
                round(policy_loss / self.update_epochs, 4),
                round(value_loss / self.update_epochs, 4),
                round((policy_loss + value_loss) / self.update_epochs, 4),
                round(kl_div / self.update_epochs, 6),
                round(entropy / self.update_epochs, 4),
                f"{current_lr:.6f}"
            ])
    
    def get_convergence_stats(self) -> Optional[Dict[str, float]]:
        """Get convergence statistics"""
        if len(self.recent_efforts) < 10:
            return None
        
        recent_mean = np.mean(self.recent_efforts[-50:]) if len(self.recent_efforts) >= 50 else np.mean(self.recent_efforts)
        recent_std = np.std(self.recent_efforts[-50:]) if len(self.recent_efforts) >= 50 else np.std(self.recent_efforts)
        
        return {
            'recent_mean_effort': recent_mean,
            'recent_std_effort': recent_std,
            'recent_mean_reward': np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else np.mean(self.recent_rewards)
        } 