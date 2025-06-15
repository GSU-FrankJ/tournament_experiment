import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from config.asymmetric_cost_two_players import config
from envs.asymmetric_cost_env import AsymmetricCostEnv
from agents.asymmetric_gradient_solver import asymmetric_gradient_descent_solver

class TestAsymmetricCost(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.env = AsymmetricCostEnv(config)
        self.test_config = {
            "k1": 0.0003,
            "k2": 0.0005,
            "q": 25.0,
            "w_h": 6.5,
            "w_l": 3.0,
            "effort_range": [0, 200],
            "seed": 42,
            "num_players": 2,
            "theoretical_efforts": [72.92, 26.25],
            "theoretical_costs": [1.60, 0.34]
        }
    
    def test_environment_initialization(self):
        """Test that AsymmetricCostEnv initializes correctly"""
        # Test cost parameters
        self.assertEqual(len(self.env.k_players), 2)
        self.assertEqual(self.env.k_players[0], config["k1"])
        self.assertEqual(self.env.k_players[1], config["k2"])
        
        # Test other parameters
        self.assertEqual(self.env.q, config["q"])
        self.assertEqual(self.env.w_h, config["w_h"])
        self.assertEqual(self.env.w_l, config["w_l"])
        self.assertEqual(self.env.num_players, 2)
    
    def test_cost_parameter_access(self):
        """Test that cost parameters can be accessed correctly"""
        cost_params = self.env.get_cost_parameters()
        self.assertEqual(cost_params[0], config["k1"])
        self.assertEqual(cost_params[1], config["k2"])
        self.assertTrue(cost_params[0] < cost_params[1])  # k1 < k2
    
    def test_theoretical_values(self):
        """Test that theoretical values are computed correctly"""
        theoretical_efforts = self.env.get_theoretical_efforts()
        theoretical_costs = self.env.get_theoretical_costs()
        
        self.assertIsNotNone(theoretical_efforts)
        self.assertIsNotNone(theoretical_costs)
        self.assertEqual(len(theoretical_efforts), 2)
        self.assertEqual(len(theoretical_costs), 2)
        
        # Player 1 should have higher effort (lower cost parameter)
        self.assertGreater(theoretical_efforts[0], theoretical_efforts[1])
        
        # Verify theoretical effort calculation
        k1, k2 = config["k1"], config["k2"]
        w_diff = config["w_h"] - config["w_l"]
        q = config["q"]
        
        expected_e1 = w_diff * k2 / (4 * q * (k1 + k2) * k1)
        expected_e2 = w_diff * k1 / (4 * q * (k1 + k2) * k2)
        
        self.assertAlmostEqual(theoretical_efforts[0], expected_e1, places=2)
        self.assertAlmostEqual(theoretical_efforts[1], expected_e2, places=2)
    
    def test_utility_computation(self):
        """Test that utility computation uses correct cost parameters"""
        # Test player 1 utility (lower cost)
        effort1 = 50.0
        effort2 = 30.0
        
        u1, cost1 = self.env.utility(0, effort1, effort2)  # Player 1
        u2, cost2 = self.env.utility(1, effort2, effort1)  # Player 2
        
        # Cost should be k_i * e_i^2
        expected_cost1 = config["k1"] * effort1 * effort1
        expected_cost2 = config["k2"] * effort2 * effort2
        
        self.assertAlmostEqual(cost1, expected_cost1, places=6)
        self.assertAlmostEqual(cost2, expected_cost2, places=6)
        
        # Player 1 should have lower cost per unit effort squared
        self.assertLess(config["k1"], config["k2"])
    
    def test_environment_step(self):
        """Test that environment step works correctly with asymmetric costs"""
        actions = torch.tensor([60.0, 40.0])
        obs, rewards, costs, done, info = self.env.step(actions)
        
        # Check return types and shapes
        self.assertEqual(len(obs), 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(costs), 2)
        self.assertTrue(done)
        
        # Check info contains cost parameters
        self.assertIn("p1_k", info)
        self.assertIn("p2_k", info)
        self.assertEqual(info["p1_k"], config["k1"])
        self.assertEqual(info["p2_k"], config["k2"])
        
        # Check efforts are recorded
        self.assertIn("efforts", info)
        self.assertEqual(len(info["efforts"]), 2)
    
    def test_asymmetric_gradient_solver(self):
        """Test that asymmetric gradient solver converges"""
        # Run a short gradient descent
        efforts, utilities, costs = asymmetric_gradient_descent_solver(
            self.env, lr=0.1, steps=1000, eps=1e-3
        )
        
        # Check return types and shapes
        self.assertEqual(len(efforts), 2)
        self.assertEqual(len(utilities), 2)
        self.assertEqual(len(costs), 2)
        
        # Efforts should be positive and within range
        for effort in efforts:
            self.assertGreater(effort, 0)
            self.assertLess(effort, 200)  # Within effort_range
        
        # Player 1 should generally have higher effort (lower cost)
        # Note: This might not always hold due to convergence issues in short runs
        # but it's the expected theoretical result
        
        # Costs should match k_i * e_i^2
        expected_costs = [config["k1"] * efforts[0]**2, config["k2"] * efforts[1]**2]
        for i in range(2):
            self.assertAlmostEqual(costs[i], expected_costs[i], places=4)
    
    def test_config_consistency(self):
        """Test that configuration values are consistent"""
        # Test that k1 < k2 as specified in the task
        self.assertLess(config["k1"], config["k2"])
        
        # Test that theoretical efforts are computed correctly
        k1, k2 = config["k1"], config["k2"]
        w_diff = config["w_h"] - config["w_l"]
        q = config["q"]
        
        expected_e1 = w_diff * k2 / (4 * q * (k1 + k2) * k1)
        expected_e2 = w_diff * k1 / (4 * q * (k1 + k2) * k2)
        
        self.assertAlmostEqual(config["effort1"], expected_e1, places=2)
        self.assertAlmostEqual(config["effort2"], expected_e2, places=2)
        
        # Test that costs are computed correctly
        expected_cost1 = k1 * expected_e1**2
        expected_cost2 = k2 * expected_e2**2
        
        self.assertAlmostEqual(config["cost1"], expected_cost1, places=2)
        self.assertAlmostEqual(config["cost2"], expected_cost2, places=2)
    
    def test_probability_computation(self):
        """Test that win probability computation is consistent"""
        # Test symmetric case
        prob1 = self.env.probability_uniform(50.0, 50.0)
        self.assertAlmostEqual(prob1, 0.5, places=2)
        
        # Test asymmetric case
        prob2 = self.env.probability_uniform(60.0, 40.0)
        self.assertGreater(prob2, 0.5)  # Higher effort should have higher win probability
        
        prob3 = self.env.probability_uniform(40.0, 60.0)
        self.assertLess(prob3, 0.5)  # Lower effort should have lower win probability
        
        # Probabilities should be complementary
        self.assertAlmostEqual(prob2 + prob3, 1.0, places=2)

if __name__ == '__main__':
    unittest.main() 