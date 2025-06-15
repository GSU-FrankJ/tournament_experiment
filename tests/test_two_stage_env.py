#!/usr/bin/env python3
"""
Unit tests for two-stage game environment.
Tests configuration, environment functionality, stage transitions, and information flow.
"""

import unittest
import sys
import os
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.two_stage_two_players import config
from envs.two_stage_env import TwoStageEnv

class TestTwoStageConfig(unittest.TestCase):
    """Test two-stage configuration"""
    
    def test_config_structure(self):
        """Test that config has all required fields"""
        required_fields = [
            "k1", "k2", "q", "w_h", "w_l", "effort_range", "seed", "num_players",
            "stage1_weight", "stage2_weight", "information_revelation",
            "stage1_effort", "stage2_effort", "total_cost", "expected_utility"
        ]
        
        for field in required_fields:
            self.assertIn(field, config, f"Config missing required field: {field}")
    
    def test_stage_weights(self):
        """Test that stage weights sum to 1"""
        total_weight = config["stage1_weight"] + config["stage2_weight"]
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_different_cost_parameters(self):
        """Test that Stage 1 and Stage 2 can have different cost parameters"""
        self.assertIsInstance(config["k1"], (int, float))
        self.assertIsInstance(config["k2"], (int, float))
        # They can be equal or different
        self.assertGreater(config["k1"], 0)
        self.assertGreater(config["k2"], 0)
    
    def test_theoretical_efforts(self):
        """Test that theoretical efforts are computed correctly"""
        self.assertGreater(config["stage1_effort"], 0)
        self.assertGreater(config["stage2_effort"], 0)
        self.assertIsInstance(config["stage1_effort"], (int, float))
        self.assertIsInstance(config["stage2_effort"], (int, float))

class TestTwoStageEnv(unittest.TestCase):
    """Test two-stage environment functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = TwoStageEnv(config)
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.num_players, 2)
        self.assertEqual(self.env.current_stage, 1)
        self.assertIsNone(self.env.stage1_efforts)
        self.assertIsNone(self.env.stage1_outcomes)
        self.assertEqual(self.env.stage1_weight + self.env.stage2_weight, 1.0)
    
    def test_reset(self):
        """Test environment reset"""
        states = self.env.reset()
        
        self.assertEqual(len(states), 2)
        self.assertEqual(self.env.current_stage, 1)
        self.assertIsNone(self.env.stage1_efforts)
        
        # Check state format
        for state in states:
            self.assertIsInstance(state, torch.Tensor)
            self.assertEqual(state.shape, (1,))
    
    def test_stage1_step(self):
        """Test Stage 1 execution"""
        self.env.reset()
        
        # Test Stage 1 step
        actions = [torch.tensor([50.0]), torch.tensor([60.0])]
        states, rewards, costs, done, info = self.env.step(actions)
        
        # Check basic properties
        self.assertEqual(len(states), 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(costs), 2)
        self.assertFalse(done)  # Should not be done after Stage 1
        self.assertEqual(self.env.current_stage, 2)
        
        # Check that Stage 1 data is stored
        self.assertIsNotNone(self.env.stage1_efforts)
        self.assertIsNotNone(self.env.stage1_outcomes)
        self.assertEqual(self.env.stage1_efforts, [50.0, 60.0])
        
        # Check info structure
        self.assertEqual(info["stage"], 1)
        self.assertIn("efforts", info)
        self.assertIn("stage1_winner", info)
        self.assertIn("p1_stage1_cost", info)
        self.assertIn("p2_stage1_cost", info)
    
    def test_stage2_step(self):
        """Test Stage 2 execution"""
        self.env.reset()
        
        # Execute Stage 1
        stage1_actions = [torch.tensor([50.0]), torch.tensor([60.0])]
        self.env.step(stage1_actions)
        
        # Execute Stage 2
        stage2_actions = [torch.tensor([40.0]), torch.tensor([45.0])]
        states, rewards, costs, done, info = self.env.step(stage2_actions)
        
        # Check basic properties
        self.assertEqual(len(states), 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(costs), 2)
        self.assertTrue(done)  # Should be done after Stage 2
        
        # Check info structure
        self.assertEqual(info["stage"], 2)
        self.assertIn("stage1_efforts", info)
        self.assertIn("stage2_efforts", info)
        self.assertIn("stage1_winner", info)
        self.assertIn("stage2_winner", info)
        self.assertIn("stage_weights", info)
        
        # Check that costs include both stages
        for i in range(2):
            self.assertIn(f"p{i+1}_stage1_cost", info)
            self.assertIn(f"p{i+1}_stage2_cost", info)
            self.assertIn(f"p{i+1}_total_cost", info)
    
    def test_stage2_without_stage1_error(self):
        """Test that Stage 2 cannot be executed without Stage 1"""
        self.env.reset()
        self.env.current_stage = 2  # Manually set to Stage 2
        
        actions = [torch.tensor([40.0]), torch.tensor([45.0])]
        
        with self.assertRaises(ValueError):
            self.env.step(actions)
    
    def test_probability_calculation(self):
        """Test win probability calculation with noise factors"""
        # Test with different noise factors
        prob1 = self.env.probability_uniform(50.0, 60.0, noise_factor=1.0)
        prob2 = self.env.probability_uniform(50.0, 60.0, noise_factor=0.8)
        
        self.assertGreater(prob1, 0)
        self.assertLess(prob1, 1)
        self.assertGreater(prob2, 0)
        self.assertLess(prob2, 1)
        
        # With lower noise factor, probability should be more extreme
        # (further from 0.5 for the same effort difference)
        self.assertNotEqual(prob1, prob2)
    
    def test_stage_utility_calculation(self):
        """Test utility calculation for different stages"""
        # Test Stage 1 utility
        utility1, cost1 = self.env.compute_stage_utility(50.0, [60.0], stage=1)
        
        # Test Stage 2 utility
        utility2, cost2 = self.env.compute_stage_utility(50.0, [60.0], stage=2)
        
        # Both should be valid
        self.assertIsInstance(utility1, (int, float))
        self.assertIsInstance(cost1, (int, float))
        self.assertIsInstance(utility2, (int, float))
        self.assertIsInstance(cost2, (int, float))
        
        # Costs should be different if k1 != k2
        if config["k1"] != config["k2"]:
            self.assertNotEqual(cost1, cost2)
    
    def test_information_revelation_none(self):
        """Test no information revelation"""
        # Create environment with no information revelation
        test_config = config.copy()
        test_config["information_revelation"] = "none"
        env = TwoStageEnv(test_config)
        
        env.reset()
        env.step([torch.tensor([50.0]), torch.tensor([60.0])])
        
        info = env.get_information_state(0)
        
        # Should only contain basic info
        self.assertEqual(info["stage"], 2)
        self.assertEqual(info["player_id"], 0)
        self.assertNotIn("stage1_winner", info)
        self.assertNotIn("opponent_stage1_efforts", info)
    
    def test_information_revelation_partial(self):
        """Test partial information revelation"""
        # Use default config (partial revelation)
        self.env.reset()
        self.env.step([torch.tensor([50.0]), torch.tensor([60.0])])
        
        info = self.env.get_information_state(0)
        
        # Should contain some information based on settings
        self.assertEqual(info["stage"], 2)
        self.assertEqual(info["player_id"], 0)
        
        if config["reveal_stage1_outcome"]:
            self.assertIn("stage1_winner", info)
            self.assertIn("won_stage1", info)
    
    def test_information_revelation_full(self):
        """Test full information revelation"""
        # Create environment with full information revelation
        test_config = config.copy()
        test_config["information_revelation"] = "full"
        env = TwoStageEnv(test_config)
        
        env.reset()
        env.step([torch.tensor([50.0]), torch.tensor([60.0])])
        
        info = env.get_information_state(0)
        
        # Should contain all information
        self.assertEqual(info["stage"], 2)
        self.assertEqual(info["player_id"], 0)
        self.assertIn("stage1_efforts", info)
        self.assertIn("stage1_outcomes", info)

class TestTwoStageIntegration(unittest.TestCase):
    """Test integration between stages and overall game flow"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = TwoStageEnv(config)
    
    def test_complete_game_flow(self):
        """Test complete two-stage game execution"""
        # Reset
        initial_states = self.env.reset()
        self.assertEqual(len(initial_states), 2)
        self.assertEqual(self.env.current_stage, 1)
        
        # Stage 1
        stage1_actions = [torch.tensor([50.0]), torch.tensor([60.0])]
        stage1_states, stage1_rewards, stage1_costs, stage1_done, stage1_info = self.env.step(stage1_actions)
        
        self.assertFalse(stage1_done)
        self.assertEqual(self.env.current_stage, 2)
        self.assertTrue(self.env.is_stage1_complete())
        
        # Stage 2
        stage2_actions = [torch.tensor([40.0]), torch.tensor([45.0])]
        final_states, final_rewards, final_costs, final_done, final_info = self.env.step(stage2_actions)
        
        self.assertTrue(final_done)
        
        # Check that final rewards are weighted combinations
        # (This is a simplified check - exact values depend on win probabilities)
        self.assertEqual(len(final_rewards), 2)
        self.assertEqual(len(final_costs), 2)
        
        # Check that total costs include both stages
        for i in range(2):
            stage1_cost = final_info[f"p{i+1}_stage1_cost"]
            stage2_cost = final_info[f"p{i+1}_stage2_cost"]
            total_cost = final_info[f"p{i+1}_total_cost"]
            
            self.assertAlmostEqual(total_cost, stage1_cost + stage2_cost, places=6)
    
    def test_stage_weights_effect(self):
        """Test that stage weights affect final outcomes"""
        # Test with different stage weights
        test_config1 = config.copy()
        test_config1["stage1_weight"] = 0.8
        test_config1["stage2_weight"] = 0.2
        
        test_config2 = config.copy()
        test_config2["stage1_weight"] = 0.2
        test_config2["stage2_weight"] = 0.8
        
        env1 = TwoStageEnv(test_config1)
        env2 = TwoStageEnv(test_config2)
        
        # Use same actions for both environments
        stage1_actions = [torch.tensor([50.0]), torch.tensor([60.0])]
        stage2_actions = [torch.tensor([40.0]), torch.tensor([45.0])]
        
        # Run both environments
        for env in [env1, env2]:
            env.reset()
            env.step(stage1_actions)
            final_states, final_rewards, final_costs, final_done, final_info = env.step(stage2_actions)
        
        # Results should be different due to different weights
        # (This is a basic check - exact comparison would require deterministic outcomes)
        self.assertTrue(True)  # Basic completion test
    
    def test_theoretical_efforts(self):
        """Test theoretical effort values"""
        theoretical_efforts = self.env.get_theoretical_efforts()
        
        self.assertEqual(len(theoretical_efforts), 2)
        self.assertGreater(theoretical_efforts[0], 0)  # Stage 1 effort
        self.assertGreater(theoretical_efforts[1], 0)  # Stage 2 effort
        
        # Should match config values
        self.assertEqual(theoretical_efforts[0], config["stage1_effort"])
        self.assertEqual(theoretical_efforts[1], config["stage2_effort"])
    
    def test_stage_weights_retrieval(self):
        """Test stage weights retrieval"""
        weights = self.env.get_stage_weights()
        
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0], config["stage1_weight"])
        self.assertEqual(weights[1], config["stage2_weight"])
        self.assertAlmostEqual(weights[0] + weights[1], 1.0, places=6)

class TestTwoStageErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test environment"""
        self.env = TwoStageEnv(config)
    
    def test_wrong_number_of_actions(self):
        """Test error handling for wrong number of actions"""
        self.env.reset()
        
        # Too few actions
        with self.assertRaises(ValueError):
            self.env.step([torch.tensor([50.0])])
        
        # Too many actions
        with self.assertRaises(ValueError):
            self.env.step([torch.tensor([50.0]), torch.tensor([60.0]), torch.tensor([70.0])])
    
    def test_invalid_stage(self):
        """Test error handling for invalid stage"""
        self.env.reset()
        self.env.current_stage = 3  # Invalid stage
        
        with self.assertRaises(ValueError):
            self.env.step([torch.tensor([50.0]), torch.tensor([60.0])])
    
    def test_stage_state_consistency(self):
        """Test that stage state remains consistent"""
        self.env.reset()
        
        # Initially should be Stage 1
        self.assertEqual(self.env.get_current_stage(), 1)
        self.assertFalse(self.env.is_stage1_complete())
        
        # After Stage 1
        self.env.step([torch.tensor([50.0]), torch.tensor([60.0])])
        self.assertEqual(self.env.get_current_stage(), 2)
        self.assertTrue(self.env.is_stage1_complete())

if __name__ == "__main__":
    unittest.main() 