#!/usr/bin/env python3
"""
Unit tests for asymmetric ability parameters experiment.
Tests configuration, environment, solver, and integration.
"""

import unittest
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.asymmetric_ability_two_players import config
from envs.asymmetric_ability_env import AsymmetricAbilityEnv
from agents.asymmetric_ability_solver import asymmetric_ability_gradient_descent_solver, verify_equilibrium_conditions

class TestAsymmetricAbilityConfig(unittest.TestCase):
    """Test asymmetric ability configuration"""
    
    def test_config_structure(self):
        """Test that config has all required fields"""
        required_fields = ["l1", "l2", "k", "q", "w_h", "w_l", "effort_range", "seed", "num_players"]
        for field in required_fields:
            self.assertIn(field, config, f"Config missing required field: {field}")
    
    def test_asymmetric_abilities(self):
        """Test that l1 > l2 (asymmetric abilities)"""
        self.assertGreater(config["l1"], config["l2"], "l1 should be greater than l2")
        self.assertGreater(config["l1"], 0, "l1 should be positive")
        self.assertGreater(config["l2"], 0, "l2 should be positive")
    
    def test_equal_cost_parameter(self):
        """Test that cost parameter k is the same for both players"""
        self.assertIn("k", config, "Config should have single cost parameter k")
        self.assertGreater(config["k"], 0, "Cost parameter k should be positive")
    
    def test_theoretical_efforts(self):
        """Test theoretical effort calculations"""
        self.assertIn("effort1", config, "Config should have theoretical effort1")
        self.assertIn("effort2", config, "Config should have theoretical effort2")
        self.assertIn("theoretical_efforts", config, "Config should have theoretical_efforts list")
        
        # Higher ability player should have higher theoretical effort
        self.assertGreater(config["effort1"], config["effort2"], 
                          "Player 1 (higher ability) should have higher theoretical effort")
        
        # Check that theoretical efforts match the formula: e_i* = (w_h - w_l) * l_i / (4 * k * q)
        expected_effort1 = (config["w_h"] - config["w_l"]) * config["l1"] / (4 * config["k"] * config["q"])
        expected_effort2 = (config["w_h"] - config["w_l"]) * config["l2"] / (4 * config["k"] * config["q"])
        
        self.assertAlmostEqual(config["effort1"], expected_effort1, places=6)
        self.assertAlmostEqual(config["effort2"], expected_effort2, places=6)

class TestAsymmetricAbilityEnv(unittest.TestCase):
    """Test asymmetric ability environment"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = AsymmetricAbilityEnv(config)
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.num_players, 2)
        self.assertEqual(len(self.env.l_players), 2)
        self.assertEqual(self.env.l_players[0], config["l1"])
        self.assertEqual(self.env.l_players[1], config["l2"])
        self.assertEqual(self.env.k, config["k"])
    
    def test_ability_parameters(self):
        """Test ability parameter access"""
        abilities = self.env.get_ability_parameters()
        self.assertEqual(len(abilities), 2)
        self.assertEqual(abilities[0], config["l1"])
        self.assertEqual(abilities[1], config["l2"])
        self.assertGreater(abilities[0], abilities[1])  # l1 > l2
    
    def test_theoretical_values(self):
        """Test theoretical values access"""
        theoretical_efforts = self.env.get_theoretical_efforts()
        theoretical_costs = self.env.get_theoretical_costs()
        
        self.assertEqual(len(theoretical_efforts), 2)
        self.assertEqual(len(theoretical_costs), 2)
        self.assertGreater(theoretical_efforts[0], theoretical_efforts[1])  # Higher ability => higher effort
    
    def test_probability_calculation(self):
        """Test win probability calculation with abilities"""
        # Test symmetric efforts with asymmetric abilities
        e1, e2 = 50.0, 50.0
        l1, l2 = self.env.l_players
        
        p1_win = self.env.probability_uniform_with_abilities(e1, e2, l1, l2)
        
        # Higher ability player should have higher win probability with equal efforts
        self.assertGreater(p1_win, 0.5, "Higher ability player should have >50% win probability with equal efforts")
        self.assertLess(p1_win, 1.0, "Win probability should be less than 1")
        self.assertGreater(p1_win, 0.0, "Win probability should be greater than 0")
    
    def test_utility_calculation(self):
        """Test utility calculation for each player"""
        e1, e2 = 100.0, 80.0
        
        # Test player 1 utility
        u1, c1 = self.env.utility(0, e1, e2)  # player_id=0 for player 1
        self.assertIsInstance(u1, float)
        self.assertIsInstance(c1, float)
        self.assertGreater(c1, 0)  # Cost should be positive
        
        # Test player 2 utility
        u2, c2 = self.env.utility(1, e2, e1)  # player_id=1 for player 2
        self.assertIsInstance(u2, float)
        self.assertIsInstance(c2, float)
        self.assertGreater(c2, 0)  # Cost should be positive
        
        # Cost should follow c = k * e^2 (same k for both players)
        expected_c1 = self.env.k * e1 * e1
        expected_c2 = self.env.k * e2 * e2
        self.assertAlmostEqual(c1, expected_c1, places=6)
        self.assertAlmostEqual(c2, expected_c2, places=6)
    
    def test_step_function(self):
        """Test environment step function"""
        actions = [torch.tensor([100.0]), torch.tensor([80.0])]
        
        obs, rewards, costs, done, info = self.env.step(actions)
        
        # Check return types and shapes
        self.assertEqual(len(obs), 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(costs), 2)
        self.assertTrue(done)
        
        # Check info dict
        self.assertIn("efforts", info)
        self.assertIn("p1_l", info)
        self.assertIn("p2_l", info)
        self.assertIn("p1_cost", info)
        self.assertIn("p2_cost", info)
        
        self.assertEqual(info["p1_l"], config["l1"])
        self.assertEqual(info["p2_l"], config["l2"])
    
    def test_win_probabilities(self):
        """Test win probability calculation"""
        # Use more balanced efforts to avoid saturation
        efforts = [70.0, 80.0]  # Lower ability player has slightly higher effort
        win_probs = self.env.get_win_probabilities(efforts)
        
        self.assertEqual(len(win_probs), 2)
        self.assertAlmostEqual(win_probs[0] + win_probs[1], 1.0, places=6)
        self.assertGreater(win_probs[0], 0)
        self.assertGreater(win_probs[1], 0)
        
        # Test with efforts that give clear advantage to higher ability player
        efforts_advantage = [90.0, 70.0]  # Higher ability player has higher effort
        win_probs_advantage = self.env.get_win_probabilities(efforts_advantage)
        self.assertGreater(win_probs_advantage[0], win_probs_advantage[1])
    
    def test_equilibrium_analysis(self):
        """Test equilibrium analysis function"""
        efforts = [105.0, 70.0]  # Close to theoretical values
        analysis = self.env.analyze_equilibrium(efforts)
        
        required_keys = ["efforts", "theoretical_efforts", "gaps", "utilities", "costs", 
                        "win_probabilities", "ability_parameters", "cost_parameter"]
        for key in required_keys:
            self.assertIn(key, analysis)
        
        self.assertEqual(len(analysis["gaps"]), 2)
        self.assertEqual(len(analysis["utilities"]), 2)
        self.assertEqual(len(analysis["costs"]), 2)
        self.assertEqual(len(analysis["win_probabilities"]), 2)

class TestAsymmetricAbilitySolver(unittest.TestCase):
    """Test asymmetric ability gradient descent solver"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = AsymmetricAbilityEnv(config)
    
    def test_solver_convergence(self):
        """Test that solver converges to reasonable values"""
        efforts, utilities, costs = asymmetric_ability_gradient_descent_solver(
            self.env, lr=0.1, steps=1000, eps=1e-3
        )
        
        # Check return types
        self.assertEqual(len(efforts), 2)
        self.assertEqual(len(utilities), 2)
        self.assertEqual(len(costs), 2)
        
        # Check that efforts are positive
        self.assertGreater(efforts[0], 0)
        self.assertGreater(efforts[1], 0)
        
        # Higher ability player should have higher effort
        self.assertGreater(efforts[0], efforts[1])
        
        # Check that costs are positive
        self.assertGreater(costs[0], 0)
        self.assertGreater(costs[1], 0)
    
    def test_equilibrium_verification(self):
        """Test equilibrium verification function"""
        # Use theoretical efforts as test case
        theoretical_efforts = self.env.get_theoretical_efforts()
        
        equilibrium_check = verify_equilibrium_conditions(self.env, theoretical_efforts)
        
        # Check return structure
        required_keys = ["gradients", "max_gradient", "is_equilibrium", "efforts"]
        for key in required_keys:
            self.assertIn(key, equilibrium_check)
        
        self.assertEqual(len(equilibrium_check["gradients"]), 2)
        self.assertEqual(len(equilibrium_check["efforts"]), 2)

class TestAsymmetricAbilityIntegration(unittest.TestCase):
    """Integration tests for asymmetric ability experiment"""
    
    def test_config_env_compatibility(self):
        """Test that config works with environment"""
        env = AsymmetricAbilityEnv(config)
        
        # Test that theoretical values are accessible
        theoretical_efforts = env.get_theoretical_efforts()
        theoretical_costs = env.get_theoretical_costs()
        
        self.assertEqual(len(theoretical_efforts), 2)
        self.assertEqual(len(theoretical_costs), 2)
        
        # Test that abilities are correctly set
        abilities = env.get_ability_parameters()
        self.assertEqual(abilities[0], config["l1"])
        self.assertEqual(abilities[1], config["l2"])
    
    def test_solver_env_compatibility(self):
        """Test that solver works with environment"""
        env = AsymmetricAbilityEnv(config)
        
        # Run a short solver test
        efforts, utilities, costs = asymmetric_ability_gradient_descent_solver(
            env, lr=0.1, steps=100, eps=1e-3
        )
        
        # Verify that solver produces valid results
        self.assertEqual(len(efforts), env.num_players)
        self.assertEqual(len(utilities), env.num_players)
        self.assertEqual(len(costs), env.num_players)
        
        # Test that results are reasonable
        for effort in efforts:
            self.assertGreater(effort, 0)
            self.assertLess(effort, 1000)  # Reasonable upper bound
    
    def test_theoretical_consistency(self):
        """Test theoretical calculations are consistent"""
        env = AsymmetricAbilityEnv(config)
        
        # Get theoretical efforts
        e1_theory, e2_theory = env.get_theoretical_efforts()
        
        # Compute utilities at theoretical efforts
        u1, c1 = env.utility(0, e1_theory, e2_theory)
        u2, c2 = env.utility(1, e2_theory, e1_theory)
        
        # At equilibrium, marginal utilities should be close to zero
        # Test this by checking small perturbations
        eps = 1e-3
        u1_plus, _ = env.utility(0, e1_theory + eps, e2_theory)
        u2_plus, _ = env.utility(1, e2_theory + eps, e1_theory)
        
        grad1 = (u1_plus - u1) / eps
        grad2 = (u2_plus - u2) / eps
        
        # Gradients should be small at equilibrium (within tolerance)
        self.assertLess(abs(grad1), 0.1, "Player 1 gradient should be small at theoretical equilibrium")
        self.assertLess(abs(grad2), 0.1, "Player 2 gradient should be small at theoretical equilibrium")

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2) 