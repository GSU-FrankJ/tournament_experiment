"""
Different Ability Environment for Two Players Tournament Game
============================================================

This environment implements a two-player contest with different ability parameters.
- Player 1: Higher ability (l1 = 10)  
- Player 2: Lower ability (l2 = 5)
- Equal cost parameters (k1 = k2 = 0.0004)

The effective effort for each player is modified by their ability parameter:
- Effective effort_i = l_i * effort_i
- Win probability depends on effective efforts plus uniform noise

Key Features:
- Proper Nash equilibrium calculation with ability scaling
- Exact win probability computation using triangular distribution
- Support for multiple q values and effort ranges
- Comprehensive analysis and logging capabilities
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class DifferentAbilityEnv:
    """
    Environment for two players with different ability parameters.
    
    This environment implements the contest model where:
    - Player 1 has ability l1 = 10 (higher)
    - Player 2 has ability l2 = 5 (lower)  
    - Both players have equal cost parameter k = 0.0004
    - Win probability depends on l_i * e_i + noise
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the different ability environment.
        
        Args:
            config: Configuration dictionary containing:
                - l1, l2: ability parameters
                - k, k1, k2: cost parameters (should be equal)
                - q: noise parameter
                - w_h, w_l: reward parameters
                - effort_range: (min, max) effort bounds
                - theoretical values for validation
        """
        # Ability parameters
        self.l1 = config.get("l1", 10)  # Player 1 ability (higher)
        self.l2 = config.get("l2", 5)   # Player 2 ability (lower)
        
        # Cost parameters (should be equal for this scenario)
        self.k = config.get("k", 0.0004)
        self.k1 = config.get("k1", self.k)  
        self.k2 = config.get("k2", self.k)
        
        # Noise and reward parameters
        self.q = config["q"]
        self.w_h = config["w_h"] 
        self.w_l = config["w_l"]
        self.w_diff = self.w_h - self.w_l
        
        # Environment settings
        self.effort_range = config["effort_range"]
        self.seed = config.get("seed", 42)
        self.num_players = 2
        
        # Theoretical values for validation
        self.theoretical_efforts = config.get("theoretical_efforts", [])
        self.theoretical_costs = config.get("theoretical_costs", [])
        self.theoretical_effort1 = config.get("theoretical_effort1", 0)
        self.theoretical_effort2 = config.get("theoretical_effort2", 0)
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"DifferentAbilityEnv initialized:")
        logger.info(f"  Abilities: l1={self.l1}, l2={self.l2}")
        logger.info(f"  Costs: k1={self.k1}, k2={self.k2}")
        logger.info(f"  Noise: q={self.q}")
        logger.info(f"  Rewards: w_h={self.w_h}, w_l={self.w_l}")
        logger.info(f"  Effort range: {self.effort_range}")
        logger.info(f"  Theoretical efforts: e1*={self.theoretical_effort1:.2f}, e2*={self.theoretical_effort2:.2f}")
    
    def _validate_config(self):
        """Validate the environment configuration."""
        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("Ability parameters must be positive")
        
        if self.k1 <= 0 or self.k2 <= 0:
            raise ValueError("Cost parameters must be positive")
            
        if self.q <= 0:
            raise ValueError("Noise parameter q must be positive")
            
        if self.w_h <= self.w_l:
            raise ValueError("High reward must be greater than low reward")
            
        if len(self.effort_range) != 2 or self.effort_range[0] >= self.effort_range[1]:
            raise ValueError("Effort range must be (min, max) with min < max")
    
    def probability_win_player1(self, e1: float, e2: float) -> float:
        """
        Calculate the probability that player 1 wins.
        
        Uses the exact formula for uniform noise:
        P(l1*e1 + ε1 > l2*e2 + ε2) where ε1, ε2 ~ Uniform(-q, q)
        
        Args:
            e1: Player 1's effort
            e2: Player 2's effort
            
        Returns:
            Probability that player 1 wins (0 to 1)
        """
        # Calculate effective efforts
        effective_e1 = self.l1 * e1
        effective_e2 = self.l2 * e2
        
        # For uniform noise model:
        # P(player 1 wins) = P(ε1 - ε2 > effective_e2 - effective_e1)
        # where ε1 - ε2 has triangular distribution over [-2q, 2q]
        
        d = effective_e2 - effective_e1
        
        # Apply the exact CDF formula for triangular distribution
        if d <= -2 * self.q:
            return 1.0
        elif d >= 2 * self.q:
            return 0.0
        elif d < 0:
            return 1.0 - ((d + 2*self.q)**2) / (8 * self.q**2)
        else:
            return ((2*self.q - d)**2) / (8 * self.q**2)
    
    def compute_utility(self, player_id: int, effort: float, other_effort: float) -> Tuple[float, float]:
        """
        Compute utility and cost for a specific player.
        
        Args:
            player_id: 0 for player 1, 1 for player 2
            effort: This player's effort
            other_effort: Other player's effort
            
        Returns:
            (utility, cost) where utility = expected_reward - cost
        """
        if player_id == 0:  # Player 1
            p_win = self.probability_win_player1(effort, other_effort)
            cost = self.k1 * effort**2
        else:  # Player 2
            p_win = 1.0 - self.probability_win_player1(other_effort, effort)
            cost = self.k2 * effort**2
        
        # Expected reward = low reward + win_probability * (high - low reward)
        expected_reward = self.w_l + p_win * self.w_diff
        utility = expected_reward - cost
        
        return utility, cost
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset the environment and return initial states.
        
        Returns:
            Tuple of initial states for both players (dummy states)
        """
        # Return dummy states (this is a single-stage game)
        state1 = torch.tensor([0.0], dtype=torch.float32)
        state2 = torch.tensor([0.0], dtype=torch.float32)
        return state1, state2
    
    def step(self, actions: List[torch.Tensor]) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], 
        torch.Tensor, 
        torch.Tensor, 
        bool, 
        Dict[str, Any]
    ]:
        """
        Execute one step with both players' actions.
        
        Args:
            actions: List of [action1, action2] as tensors
            
        Returns:
            (next_states, rewards, costs, done, info)
        """
        if len(actions) != 2:
            raise ValueError(f"Expected 2 actions, got {len(actions)}")
        
        # Extract efforts from actions
        e1 = float(actions[0].item())
        e2 = float(actions[1].item())
        
        # Clamp efforts to valid range
        e1 = max(self.effort_range[0], min(self.effort_range[1], e1))
        e2 = max(self.effort_range[0], min(self.effort_range[1], e2))
        
        # Compute utilities and costs
        utility1, cost1 = self.compute_utility(0, e1, e2)
        utility2, cost2 = self.compute_utility(1, e2, e1)
        
        # Calculate win probability for analysis
        p1_win = self.probability_win_player1(e1, e2)
        
        # Package results
        next_states = (torch.tensor([0.0]), torch.tensor([0.0]))  # Dummy states
        rewards = torch.tensor([utility1, utility2], dtype=torch.float32)
        costs = torch.tensor([cost1, cost2], dtype=torch.float32)
        done = True  # Single-stage game
        
        info = {
            "efforts": [e1, e2],
            "effective_efforts": [self.l1 * e1, self.l2 * e2],
            "win_probabilities": [p1_win, 1.0 - p1_win],
            "costs": [cost1, cost2],
            "utilities": [utility1, utility2],
            "ability_parameters": [self.l1, self.l2],
            "cost_parameters": [self.k1, self.k2]
        }
        
        logger.debug(f"Step: e1={e1:.2f}, e2={e2:.2f}, p1_win={p1_win:.3f}, u1={utility1:.3f}, u2={utility2:.3f}")
        
        return next_states, rewards, costs, done, info
    
    def analyze_equilibrium(self, efforts: List[float]) -> Dict[str, Any]:
        """
        Analyze how close the given efforts are to Nash equilibrium.
        
        Args:
            efforts: [e1, e2] effort values
            
        Returns:
            Comprehensive analysis dictionary
        """
        if len(efforts) != 2:
            raise ValueError("Expected 2 efforts for analysis")
        
        e1, e2 = efforts
        
        # Compute current utilities and costs
        utility1, cost1 = self.compute_utility(0, e1, e2)
        utility2, cost2 = self.compute_utility(1, e2, e1)
        
        # Win probabilities
        p1_win = self.probability_win_player1(e1, e2)
        p2_win = 1.0 - p1_win
        
        # Gaps from theoretical values
        gaps = []
        if self.theoretical_efforts and len(self.theoretical_efforts) == 2:
            gaps = [abs(e1 - self.theoretical_efforts[0]), abs(e2 - self.theoretical_efforts[1])]
        elif self.theoretical_effort1 > 0 and self.theoretical_effort2 > 0:
            gaps = [abs(e1 - self.theoretical_effort1), abs(e2 - self.theoretical_effort2)]
        
        # Effective efforts for ability comparison
        eff_e1 = self.l1 * e1
        eff_e2 = self.l2 * e2
        
        analysis = {
            "efforts": efforts,
            "effective_efforts": [eff_e1, eff_e2],
            "theoretical_efforts": self.theoretical_efforts or [self.theoretical_effort1, self.theoretical_effort2],
            "gaps": gaps,
            "max_gap": max(gaps) if gaps else 0,
            "avg_gap": sum(gaps) / len(gaps) if gaps else 0,
            "utilities": [utility1, utility2],
            "costs": [cost1, cost2],
            "win_probabilities": [p1_win, p2_win],
            "ability_parameters": [self.l1, self.l2],
            "cost_parameters": [self.k1, self.k2],
            "ability_advantage": (eff_e1 - eff_e2),  # How much player 1's effective effort exceeds player 2's
            "win_advantage": (p1_win - p2_win)       # How much player 1's win prob exceeds player 2's
        }
        
        # Quality assessment
        if gaps:
            max_gap = max(gaps)
            if max_gap < 0.5:
                quality = "Excellent"
            elif max_gap < 1.0:
                quality = "Good"  
            elif max_gap < 5.0:
                quality = "Fair"
            else:
                quality = "Poor"
            analysis["convergence_quality"] = quality
        
        logger.info(f"Equilibrium analysis: gaps={gaps}, quality={analysis.get('convergence_quality', 'Unknown')}")
        
        return analysis
    
    def compute_gradients(self, efforts: List[float], eps: float = 1e-4) -> List[float]:
        """
        Compute numerical gradients of utility functions.
        
        For Nash equilibrium, both gradients should be close to zero.
        
        Args:
            efforts: [e1, e2] current effort values
            eps: Small epsilon for finite differences
            
        Returns:
            [grad1, grad2] gradients of utility w.r.t. own effort
        """
        e1, e2 = efforts
        gradients = []
        
        # Gradient for player 1
        u1_current, _ = self.compute_utility(0, e1, e2)
        u1_plus, _ = self.compute_utility(0, e1 + eps, e2)
        grad1 = (u1_plus - u1_current) / eps
        gradients.append(grad1)
        
        # Gradient for player 2  
        u2_current, _ = self.compute_utility(1, e2, e1)
        u2_plus, _ = self.compute_utility(1, e2 + eps, e1)
        grad2 = (u2_plus - u2_current) / eps
        gradients.append(grad2)
        
        return gradients
    
    def get_theoretical_efforts(self) -> List[float]:
        """Get theoretical optimal efforts."""
        if self.theoretical_efforts:
            return self.theoretical_efforts.copy()
        return [self.theoretical_effort1, self.theoretical_effort2]
    
    def get_ability_parameters(self) -> List[float]:
        """Get ability parameters."""
        return [self.l1, self.l2]
    
    def get_cost_parameters(self) -> List[float]:
        """Get cost parameters."""
        return [self.k1, self.k2] 