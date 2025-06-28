import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_ppo_agent import (
    EnhancedPPOAgent, ContinuousActionSpace, DiscreteActionSpace,
    ContinuousPolicyNetwork, DiscretePolicyNetwork, ValueNetwork,
    GAECalculator, PPOLoss
)

class TestActionSpaces(unittest.TestCase):
    """Test action space definitions"""
    
    def test_continuous_action_space(self):
        """Test continuous action space creation"""
        action_space = ContinuousActionSpace(low=0.0, high=100.0, shape=(1,))
        
        self.assertEqual(action_space.low, 0.0)
        self.assertEqual(action_space.high, 100.0)
        self.assertEqual(action_space.shape, (1,))
        self.assertEqual(action_space.action_type, 'continuous')
        
    def test_discrete_action_space(self):
        """Test discrete action space creation"""
        action_space = DiscreteActionSpace(n=10)
        
        self.assertEqual(action_space.n, 10)
        self.assertEqual(action_space.action_type, 'discrete')

class TestPolicyNetworks(unittest.TestCase):
    """Test policy network implementations"""
    
    def test_continuous_policy_network(self):
        """Test continuous policy network"""
        network = ContinuousPolicyNetwork(
            input_dim=1, hidden_dim=64, num_layers=2,
            activation='tanh', action_dim=1
        )
        
        # Test forward pass
        x = torch.randn(5, 1)
        mean, std = network(x)
        
        self.assertEqual(mean.shape, (5, 1))
        self.assertEqual(std.shape, (5, 1))
        self.assertTrue(torch.all(mean >= 0) and torch.all(mean <= 1))
        self.assertTrue(torch.all(std > 0))
        
        # Test action distribution
        dist = network.get_action_distribution(x)
        self.assertIsInstance(dist, torch.distributions.Normal)
        
    def test_discrete_policy_network(self):
        """Test discrete policy network"""
        network = DiscretePolicyNetwork(
            input_dim=1, hidden_dim=64, num_layers=2,
            activation='relu', num_actions=10
        )
        
        # Test forward pass
        x = torch.randn(5, 1)
        logits = network(x)
        
        self.assertEqual(logits.shape, (5, 10))
        
        # Test action distribution
        dist = network.get_action_distribution(x)
        self.assertIsInstance(dist, torch.distributions.Categorical)
        
    def test_value_network(self):
        """Test value network"""
        network = ValueNetwork(
            input_dim=1, hidden_dim=64, num_layers=2,
            activation='elu'
        )
        
        # Test forward pass
        x = torch.randn(5, 1)
        value = network(x)
        
        self.assertEqual(value.shape, (5, 1))

class TestGAECalculator(unittest.TestCase):
    """Test Generalized Advantage Estimation"""
    
    def test_gae_computation(self):
        """Test GAE computation"""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
        
        advantages, returns = GAECalculator.compute_gae(
            rewards, values, gamma=0.99, gae_lambda=0.95
        )
        
        self.assertEqual(len(advantages), len(rewards))
        self.assertEqual(len(returns), len(rewards))
        self.assertTrue(torch.all(torch.isfinite(advantages)))
        self.assertTrue(torch.all(torch.isfinite(returns)))

class TestPPOLoss(unittest.TestCase):
    """Test PPO loss computation"""
    
    def test_policy_loss(self):
        """Test policy loss computation"""
        new_log_probs = torch.tensor([0.1, 0.2, 0.3])
        old_log_probs = torch.tensor([0.15, 0.18, 0.25])
        advantages = torch.tensor([1.0, -0.5, 2.0])
        
        loss = PPOLoss.compute_policy_loss(
            new_log_probs, old_log_probs, advantages, clip_epsilon=0.2
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        
    def test_value_loss(self):
        """Test value loss computation"""
        new_values = torch.tensor([1.0, 2.0, 3.0])
        old_values = torch.tensor([0.9, 2.1, 2.8])
        returns = torch.tensor([1.1, 1.9, 3.2])
        
        loss = PPOLoss.compute_value_loss(
            new_values, old_values, returns, clip_epsilon=0.2
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

class TestEnhancedPPOAgentContinuous(unittest.TestCase):
    """Test Enhanced PPO Agent with continuous action space"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.action_space = ContinuousActionSpace(low=0.0, high=100.0)
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_log.csv")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        os.rmdir(self.temp_dir)
        
    def test_agent_initialization_continuous(self):
        """Test agent initialization with continuous action space"""
        agent = EnhancedPPOAgent(
            action_space=self.action_space,
            lr=1e-3,
            hidden_dim=64,
            separate_networks=True,
            log_path=self.log_path
        )
        
        self.assertEqual(agent.action_space.action_type, 'continuous')
        self.assertIsInstance(agent.policy_network, ContinuousPolicyNetwork)
        self.assertIsInstance(agent.value_network, ValueNetwork)
        self.assertTrue(agent.separate_networks)
        
    def test_action_selection_continuous(self):
        """Test action selection with continuous action space"""
        agent = EnhancedPPOAgent(action_space=self.action_space)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Test single action selection
        action = agent.select_action(state)
        
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(0 <= action.item() <= 100)
        
        # Test trajectory storage
        self.assertEqual(len(agent.states), 1)
        self.assertEqual(len(agent.actions), 1)
        self.assertEqual(len(agent.log_probs), 1)
        self.assertEqual(len(agent.values), 1)
        
    def test_policy_update_continuous(self):
        """Test policy update with continuous action space"""
        agent = EnhancedPPOAgent(action_space=self.action_space, lr=1e-3)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate trajectory
        action = agent.select_action(state)
        agent.store_reward(torch.tensor(1.0))
        
        # Update policy
        losses = agent.update_policy(episode=1)
        
        # Check that losses are returned
        self.assertIn('policy_loss', losses)
        self.assertIn('value_loss', losses)
        self.assertIn('entropy', losses)
        self.assertIn('kl_div', losses)
        
        # Trajectory should be cleared
        self.assertEqual(len(agent.states), 0)

class TestEnhancedPPOAgentDiscrete(unittest.TestCase):
    """Test Enhanced PPO Agent with discrete action space"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.action_space = DiscreteActionSpace(n=10)
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_log.csv")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        os.rmdir(self.temp_dir)
        
    def test_agent_initialization_discrete(self):
        """Test agent initialization with discrete action space"""
        agent = EnhancedPPOAgent(
            action_space=self.action_space,
            lr=1e-3,
            hidden_dim=64,
            separate_networks=False,
            log_path=self.log_path
        )
        
        self.assertEqual(agent.action_space.action_type, 'discrete')
        self.assertIsInstance(agent.policy_network, DiscretePolicyNetwork)
        self.assertIsInstance(agent.value_network, ValueNetwork)
        self.assertFalse(agent.separate_networks)
        
    def test_action_selection_discrete(self):
        """Test action selection with discrete action space"""
        agent = EnhancedPPOAgent(action_space=self.action_space)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Test single action selection
        action = agent.select_action(state)
        
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, ())  # Scalar for discrete
        self.assertTrue(0 <= action.item() < 10)
        
        # Test trajectory storage
        self.assertEqual(len(agent.states), 1)
        self.assertEqual(len(agent.actions), 1)
        self.assertEqual(len(agent.log_probs), 1)
        self.assertEqual(len(agent.values), 1)
        
    def test_policy_update_discrete(self):
        """Test policy update with discrete action space"""
        agent = EnhancedPPOAgent(action_space=self.action_space, lr=1e-3)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate trajectory
        action = agent.select_action(state)
        agent.store_reward(torch.tensor(1.0))
        
        # Update policy
        losses = agent.update_policy(episode=1)
        
        # Check that losses are returned
        self.assertIn('policy_loss', losses)
        self.assertIn('value_loss', losses)
        self.assertIn('entropy', losses)
        self.assertIn('kl_div', losses)
        
        # Trajectory should be cleared
        self.assertEqual(len(agent.states), 0)

class TestEnhancedPPOAgentFeatures(unittest.TestCase):
    """Test advanced features of Enhanced PPO Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.continuous_space = ContinuousActionSpace(low=0.0, high=100.0)
        self.discrete_space = DiscreteActionSpace(n=10)
        
    def test_reward_normalization(self):
        """Test reward normalization feature"""
        agent = EnhancedPPOAgent(
            action_space=self.continuous_space,
            reward_normalization=True
        )
        
        # Store rewards
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        for reward in rewards:
            agent.store_reward(torch.tensor(reward))
            
        # Check normalization statistics
        self.assertGreater(agent.reward_mean, 0)
        self.assertGreater(agent.reward_std, 0)
        
    def test_learning_rate_schedulers(self):
        """Test different learning rate schedulers"""
        schedulers = ['constant', 'cosine_annealing', 'step']
        
        for scheduler_type in schedulers:
            agent = EnhancedPPOAgent(
                action_space=self.continuous_space,
                lr_schedule=scheduler_type,
                separate_networks=True
            )
            
            if scheduler_type == 'constant':
                self.assertIsNone(agent.policy_scheduler)
                self.assertIsNone(agent.value_scheduler)
            else:
                self.assertIsNotNone(agent.policy_scheduler)
                self.assertIsNotNone(agent.value_scheduler)
                
    def test_separate_vs_shared_networks(self):
        """Test separate vs shared network configurations"""
        # Separate networks
        agent_separate = EnhancedPPOAgent(
            action_space=self.continuous_space,
            separate_networks=True
        )
        
        self.assertTrue(hasattr(agent_separate, 'policy_optimizer'))
        self.assertTrue(hasattr(agent_separate, 'value_optimizer'))
        
        # Shared networks
        agent_shared = EnhancedPPOAgent(
            action_space=self.continuous_space,
            separate_networks=False
        )
        
        self.assertTrue(hasattr(agent_shared, 'optimizer'))
        
    def test_convergence_statistics(self):
        """Test convergence statistics computation"""
        agent = EnhancedPPOAgent(action_space=self.continuous_space)
        
        # Initially no stats
        stats = agent.get_convergence_stats()
        self.assertIsNone(stats)
        
        # Add data
        for i in range(20):
            agent.recent_efforts.append(80.0 + i)
            agent.recent_rewards.append(1.0 + i * 0.1)
            
        stats = agent.get_convergence_stats()
        self.assertIsNotNone(stats)
        self.assertIn('recent_mean_effort', stats)
        self.assertIn('recent_std_effort', stats)
        self.assertIn('recent_mean_reward', stats)
        
    def test_logging_functionality(self):
        """Test logging functionality"""
        temp_dir = tempfile.mkdtemp()
        log_path = os.path.join(temp_dir, "test_log.csv")
        
        try:
            agent = EnhancedPPOAgent(
                action_space=self.continuous_space,
                log_path=log_path
            )
            
            # Generate training data
            state = torch.tensor([[0.0]], dtype=torch.float32)
            action = agent.select_action(state)
            agent.store_reward(torch.tensor(1.0))
            agent.update_policy(episode=1)
            
            # Check log file
            self.assertTrue(os.path.exists(log_path))
            
            with open(log_path, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 1)  # Header + data
                
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
            os.rmdir(temp_dir)
            
    def test_multi_step_trajectory(self):
        """Test multi-step trajectory handling"""
        agent = EnhancedPPOAgent(
            action_space=self.continuous_space,
            gae_lambda=0.95
        )
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate multi-step trajectory
        for i in range(10):
            action = agent.select_action(state)
            agent.store_reward(torch.tensor(float(i + 1)))
            
        # Update should handle multi-step trajectory
        losses = agent.update_policy(episode=1)
        
        # Should complete without errors
        self.assertEqual(len(agent.states), 0)
        self.assertIn('policy_loss', losses)

class TestEnhancedPPOAgentIntegration(unittest.TestCase):
    """Integration tests for Enhanced PPO Agent"""
    
    def test_continuous_learning_task(self):
        """Test learning on a simple continuous task"""
        action_space = ContinuousActionSpace(low=0.0, high=100.0)
        agent = EnhancedPPOAgent(
            action_space=action_space,
            lr=1e-3,
            entropy_coef=0.01
        )
        
        # Simple task: get close to target value
        target = 75.0
        rewards = []
        
        for episode in range(20):
            state = torch.tensor([[0.0]], dtype=torch.float32)
            action = agent.select_action(state)
            
            # Reward based on distance to target
            reward = -abs(action.item() - target) / 100.0
            agent.store_reward(torch.tensor(reward))
            agent.update_policy(episode=episode)
            rewards.append(reward)
            
        # Should show some learning
        early_rewards = np.mean(rewards[:5])
        late_rewards = np.mean(rewards[-5:])
        
        # Allow for variance but expect general improvement
        self.assertGreaterEqual(late_rewards, early_rewards - 0.1)
        
    def test_discrete_learning_task(self):
        """Test learning on a simple discrete task"""
        action_space = DiscreteActionSpace(n=10)
        agent = EnhancedPPOAgent(
            action_space=action_space,
            lr=1e-3,
            entropy_coef=0.01
        )
        
        # Simple task: prefer higher action indices
        rewards = []
        
        for episode in range(20):
            state = torch.tensor([[0.0]], dtype=torch.float32)
            action = agent.select_action(state)
            
            # Reward increases with action index
            reward = action.item() / 10.0
            agent.store_reward(torch.tensor(reward))
            agent.update_policy(episode=episode)
            rewards.append(reward)
            
        # Should show some learning
        early_rewards = np.mean(rewards[:5])
        late_rewards = np.mean(rewards[-5:])
        
        # Allow for variance but expect general improvement
        self.assertGreaterEqual(late_rewards, early_rewards - 0.1)
        
    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        agent = EnhancedPPOAgent(
            action_space=ContinuousActionSpace(low=0.0, high=100.0),
            max_grad_norm=0.5
        )
        
        state = torch.tensor([[0.0]], dtype=torch.float32)
        action = agent.select_action(state)
        agent.store_reward(torch.tensor(1000.0))  # Extreme reward
        
        # Should not raise errors due to gradient explosion
        losses = agent.update_policy(episode=1)
        
        # Check that gradients are reasonable
        for param in agent.policy_network.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertLessEqual(grad_norm, 10.0)  # Should be reasonable

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 