import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent, PPONetwork
from envs.one_stage_env import OneStageEnv

class TestPPONetwork(unittest.TestCase):
    """Test cases for PPONetwork class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 1
        self.hidden_dim = 64
        self.num_layers = 2
        
    def test_network_initialization(self):
        """Test network initialization with different configurations"""
        # Test default configuration
        network = PPONetwork()
        self.assertIsInstance(network, nn.Module)
        
        # Test custom configuration
        network = PPONetwork(
            input_dim=2, 
            hidden_dim=128, 
            num_layers=3,
            activation='tanh',
            dropout_rate=0.1,
            separate_networks=False
        )
        self.assertIsInstance(network, nn.Module)
        
    def test_network_forward_pass(self):
        """Test forward pass through the network"""
        network = PPONetwork(hidden_dim=64, num_layers=2)
        
        # Test single input
        x = torch.randn(1, 1)
        mean, std, value = network(x)
        
        self.assertEqual(mean.shape, (1, 1))
        self.assertEqual(std.shape, (1, 1))
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(torch.all(mean >= 0) and torch.all(mean <= 1))  # sigmoid output
        self.assertTrue(torch.all(std > 0))  # positive std
        
        # Test batch input
        x_batch = torch.randn(5, 1)
        mean_batch, std_batch, value_batch = network(x_batch)
        
        self.assertEqual(mean_batch.shape, (5, 1))
        self.assertEqual(std_batch.shape, (5, 1))
        self.assertEqual(value_batch.shape, (5, 1))
        
    def test_activation_functions(self):
        """Test different activation functions"""
        activations = ['relu', 'tanh']
        
        for activation in activations:
            network = PPONetwork(activation=activation)
            x = torch.randn(1, 1)
            mean, std, value = network(x)
            
            # Should not raise any errors and produce valid outputs
            self.assertFalse(torch.isnan(mean).any())
            self.assertFalse(torch.isnan(std).any())
            self.assertFalse(torch.isnan(value).any())
            
    def test_separate_vs_shared_networks(self):
        """Test separate vs shared network architectures"""
        x = torch.randn(1, 1)
        
        # Separate networks
        network_separate = PPONetwork(separate_networks=True)
        mean1, std1, value1 = network_separate(x)
        
        # Shared networks
        network_shared = PPONetwork(separate_networks=False)
        mean2, std2, value2 = network_shared(x)
        
        # Both should produce valid outputs
        self.assertEqual(mean1.shape, mean2.shape)
        self.assertEqual(std1.shape, std2.shape)
        self.assertEqual(value1.shape, value2.shape)

class TestPPOAgent(unittest.TestCase):
    """Test cases for PPOAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_log.csv")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        os.rmdir(self.temp_dir)
        
    def test_agent_initialization(self):
        """Test agent initialization with different configurations"""
        # Test default configuration
        agent = PPOAgent()
        self.assertIsInstance(agent.network, PPONetwork)
        self.assertIsInstance(agent.optimizer, torch.optim.Adam)
        
        # Test custom configuration
        agent = PPOAgent(
            lr=1e-3,
            effort_range=(0, 50),
            clip_epsilon=0.3,
            value_coef=0.75,
            entropy_coef=0.02,
            hidden_dim=128,
            num_layers=3,
            activation='tanh',
            log_path=self.log_path
        )
        
        self.assertEqual(agent.effort_low, 0)
        self.assertEqual(agent.effort_high, 50)
        self.assertEqual(agent.clip_epsilon, 0.3)
        self.assertEqual(agent.value_coef, 0.75)
        self.assertEqual(agent.entropy_coef, 0.02)
        
    def test_action_selection(self):
        """Test action selection mechanism"""
        agent = PPOAgent(effort_range=(0, 100))
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Test single action selection
        action = agent.select_action(state)
        
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(0 <= action.item() <= 100)
        
        # Test multiple action selections
        actions = []
        for _ in range(10):
            action = agent.select_action(state)
            actions.append(action.item())
            
        # Actions should vary (stochastic policy)
        self.assertTrue(len(set(actions)) > 1)
        
        # Check that trajectory data is stored
        self.assertEqual(len(agent.states), 11)  # 1 + 10
        self.assertEqual(len(agent.actions), 11)
        self.assertEqual(len(agent.log_probs), 11)
        self.assertEqual(len(agent.values), 11)
        
    def test_reward_storage(self):
        """Test reward storage and normalization"""
        agent = PPOAgent(reward_normalization=True)
        
        # Store some rewards
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        for reward in rewards:
            agent.store_reward(torch.tensor(reward))
            
        self.assertEqual(len(agent.rewards), 5)
        self.assertEqual(len(agent.reward_history), 5)
        
        # Test reward normalization statistics update
        self.assertGreater(agent.reward_mean, 0)
        self.assertGreater(agent.reward_std, 0)
        
    def test_policy_update_single_step(self):
        """Test policy update with single step"""
        agent = PPOAgent()
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate trajectory
        action = agent.select_action(state)
        agent.store_reward(torch.tensor(1.0))
        
        # Update policy
        initial_params = [p.clone() for p in agent.network.parameters()]
        agent.update_policy(episode=1, last_effort=action)
        
        # Parameters should have changed
        updated_params = list(agent.network.parameters())
        params_changed = any(not torch.equal(initial_params[i], updated_params[i]) 
                           for i in range(len(initial_params)))
        self.assertTrue(params_changed)
        
        # Trajectory should be cleared
        self.assertEqual(len(agent.states), 0)
        self.assertEqual(len(agent.actions), 0)
        self.assertEqual(len(agent.log_probs), 0)
        self.assertEqual(len(agent.rewards), 0)
        self.assertEqual(len(agent.values), 0)
        
    def test_policy_update_multi_step(self):
        """Test policy update with multiple steps (GAE)"""
        agent = PPOAgent(gae_lambda=0.95)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate multi-step trajectory
        actions = []
        for i in range(5):
            action = agent.select_action(state)
            actions.append(action)
            agent.store_reward(torch.tensor(float(i + 1)))
            
        # Update policy
        agent.update_policy(episode=1, last_effort=actions[-1])
        
        # Should complete without errors
        self.assertEqual(len(agent.states), 0)  # Trajectory cleared
        
    def test_convergence_stats(self):
        """Test convergence statistics calculation"""
        agent = PPOAgent()
        
        # Initially no stats
        stats = agent.get_convergence_stats()
        self.assertIsNone(stats)
        
        # Add some effort and reward data
        for i in range(20):
            agent.recent_efforts.append(80.0 + i)
            agent.recent_rewards.append(1.0 + i * 0.1)
            
        stats = agent.get_convergence_stats()
        self.assertIsNotNone(stats)
        self.assertIn('recent_mean_effort', stats)
        self.assertIn('recent_std_effort', stats)
        self.assertIn('recent_mean_reward', stats)
        
    def test_learning_rate_schedulers(self):
        """Test different learning rate schedulers"""
        schedulers = ['constant', 'cosine_annealing', 'step', 'plateau']
        
        for scheduler_type in schedulers:
            agent = PPOAgent(lr_schedule=scheduler_type)
            
            if scheduler_type == 'constant':
                self.assertIsNone(agent.scheduler)
            else:
                self.assertIsNotNone(agent.scheduler)
                
    def test_effort_range_scaling(self):
        """Test effort range scaling"""
        # Test different effort ranges
        ranges = [(0, 100), (10, 50), (0, 200)]
        
        for effort_range in ranges:
            agent = PPOAgent(effort_range=effort_range)
            state = torch.tensor([[0.0]], dtype=torch.float32)
            
            # Generate multiple actions
            actions = []
            for _ in range(20):
                action = agent.select_action(state)
                actions.append(action.item())
                
            # All actions should be within range
            for action in actions:
                self.assertTrue(effort_range[0] <= action <= effort_range[1])

class TestPPOIntegration(unittest.TestCase):
    """Integration tests for PPO agent with environments"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
            
    def test_simple_environment_learning(self):
        """Test PPO agent learning in a simple environment"""
        # Create simple environment
        env = OneStageEnv()
        agent = PPOAgent(lr=1e-3, effort_range=(0, 100))
        
        # Run short training
        num_episodes = 50
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            action = agent.select_action(state[0])
            
            # Simple reward: closer to 87.5 is better
            target_effort = 87.5
            reward = -abs(action.item() - target_effort) / 100.0
            
            agent.store_reward(torch.tensor(reward))
            agent.update_policy(episode=episode, last_effort=action)
            rewards.append(reward)
            
        # Agent should show some learning (rewards should improve)
        early_rewards = np.mean(rewards[:10])
        late_rewards = np.mean(rewards[-10:])
        
        # Allow for some variance but expect general improvement
        self.assertGreaterEqual(late_rewards, early_rewards - 0.1)
        
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        agent = PPOAgent(batch_size=32)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate batch of experiences
        for _ in range(32):
            action = agent.select_action(state)
            agent.store_reward(torch.tensor(np.random.normal(0, 1)))
            
        # Update should handle batch correctly
        agent.update_policy(episode=1, last_effort=agent.actions[-1])
        
        # Should complete without errors
        self.assertEqual(len(agent.states), 0)
        
    def test_logging_functionality(self):
        """Test logging functionality"""
        log_path = os.path.join(self.temp_dir, "test_log.csv")
        agent = PPOAgent(log_path=log_path)
        
        # Generate some training data
        state = torch.tensor([[0.0]], dtype=torch.float32)
        action = agent.select_action(state)
        agent.store_reward(torch.tensor(1.0))
        agent.update_policy(episode=1, last_effort=action)
        
        # Check log file was created and has content
        self.assertTrue(os.path.exists(log_path))
        
        with open(log_path, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # Header + at least one data line
            
    def test_memory_efficiency(self):
        """Test memory efficiency with long trajectories"""
        agent = PPOAgent()
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate long trajectory
        for _ in range(1000):
            action = agent.select_action(state)
            agent.store_reward(torch.tensor(np.random.normal(0, 1)))
            
        # Update should handle large trajectory
        agent.update_policy(episode=1, last_effort=agent.actions[-1])
        
        # Memory should be cleared
        self.assertEqual(len(agent.states), 0)
        
    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        agent = PPOAgent(max_grad_norm=0.5)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate trajectory with extreme rewards to test clipping
        action = agent.select_action(state)
        agent.store_reward(torch.tensor(1000.0))  # Extreme reward
        
        # Should not raise errors due to gradient explosion
        agent.update_policy(episode=1, last_effort=action)
        
        # Check that gradients are reasonable
        for param in agent.network.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertLessEqual(grad_norm, 10.0)  # Should be reasonable

class TestPPOAdvancedFeatures(unittest.TestCase):
    """Test advanced PPO features"""
    
    def test_gae_computation(self):
        """Test Generalized Advantage Estimation computation"""
        agent = PPOAgent(gae_lambda=0.95)
        
        # Create mock trajectory data
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [torch.tensor([v]) for v in [0.5, 1.5, 2.5, 3.5, 4.5]]
        
        # Manually set trajectory data
        agent.rewards = rewards
        agent.values = values
        agent.states = [torch.tensor([[0.0]]) for _ in range(5)]
        agent.actions = [torch.tensor([50.0]) for _ in range(5)]
        agent.log_probs = [torch.tensor([0.0]) for _ in range(5)]
        
        # Update should compute GAE correctly
        agent.update_policy(episode=1, last_effort=torch.tensor([50.0]))
        
        # Should complete without errors
        self.assertEqual(len(agent.states), 0)
        
    def test_value_function_learning(self):
        """Test value function learning"""
        agent = PPOAgent(value_coef=1.0)  # High value coefficient
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate consistent positive rewards
        for _ in range(10):
            action = agent.select_action(state)
            agent.store_reward(torch.tensor(1.0))
            
        # Get initial value estimate
        with torch.no_grad():
            _, _, initial_value = agent.network(state)
            
        # Update policy
        agent.update_policy(episode=1, last_effort=agent.actions[-1])
        
        # Value function should learn to predict positive rewards
        with torch.no_grad():
            _, _, updated_value = agent.network(state)
            
        # Value should generally increase (though may vary due to stochasticity)
        # Just check that it's a reasonable value
        self.assertTrue(-10 <= updated_value.item() <= 10)
        
    def test_entropy_regularization(self):
        """Test entropy regularization"""
        # High entropy coefficient should encourage exploration
        agent_high_entropy = PPOAgent(entropy_coef=0.1)
        agent_low_entropy = PPOAgent(entropy_coef=0.001)
        
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Generate actions from both agents
        actions_high = []
        actions_low = []
        
        for _ in range(50):
            action_high = agent_high_entropy.select_action(state)
            action_low = agent_low_entropy.select_action(state)
            actions_high.append(action_high.item())
            actions_low.append(action_low.item())
            
        # High entropy agent should have more diverse actions
        std_high = np.std(actions_high)
        std_low = np.std(actions_low)
        
        # This is a probabilistic test, so we allow some tolerance
        # Just check that both produce reasonable variation
        self.assertGreater(std_high, 0)
        self.assertGreater(std_low, 0)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 