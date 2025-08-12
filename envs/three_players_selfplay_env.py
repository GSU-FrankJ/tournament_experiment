#!/usr/bin/env python3
"""
Three Players Self-Play Environment
===================================

This environment supports true self-play for three players where each player
can independently choose their effort levels without being forced into symmetric equilibrium.

Key Features:
- Each player has their own independent action space
- No assumption of symmetric equilibrium
- True self-play where players learn optimal strategies independently
- Support for asymmetric equilibrium discovery
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from collections import deque

class ThreePlayersSelfPlayEnv:
    """
    Three players tournament environment with true self-play support.
    
    Each player can independently choose their effort level, and the environment
    will compute win probabilities and utilities based on the actual effort choices
    without assuming symmetry.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the three players self-play environment.
        
        Args:
            config: Configuration dictionary containing:
                - k: Cost parameter
                - q: Noise parameter  
                - w_h: High reward
                - w_l: Low reward
                - effort_range: (min_effort, max_effort)
                - seed: Random seed
                - num_players: Number of players (should be 3)
        """
        self.k = config["k"]
        self.q = config["q"]
        self.w_h = config["w_h"]
        self.w_l = config["w_l"]
        self.effort_range = config["effort_range"]
        self.seed = config["seed"]
        self.num_players = config.get("num_players", 3)
        
        # Validate number of players
        if self.num_players != 3:
            raise ValueError(f"ThreePlayersSelfPlayEnv only supports 3 players, got {self.num_players}")
        
        # Theoretical values (for reference only, not used in self-play)
        self.e_star = config.get("effort", None)
        self.cost_star = config.get("cost", None)
        self.eu_star = config.get("eu", None)
        
        # Environment state
        self.current_episode = 0
        self.episode_history = deque(maxlen=1000)
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Log environment setup
        print(f"🎮 ThreePlayersSelfPlayEnv initialized:")
        print(f"   - Cost parameter (k): {self.k}")
        print(f"   - Noise parameter (q): {self.q}")
        print(f"   - Rewards: w_h={self.w_h}, w_l={self.w_l}")
        print(f"   - Effort range: {self.effort_range}")
        print(f"   - Theoretical effort: {self.e_star:.3f}" if self.e_star else "   - No theoretical effort provided")
    
    def probability_win_three_players(self, e1: float, e2: float, e3: float) -> float:
        """
        Calculate win probability for player 1 given efforts of all three players.
        
        This uses Monte Carlo simulation to accurately compute win probabilities
        for any combination of effort levels, not just symmetric cases.
        
        Args:
            e1: Effort of player 1 (the player we're calculating for)
            e2: Effort of player 2
            e3: Effort of player 3
            
        Returns:
            Probability that player 1 wins
        """
        # Use Monte Carlo simulation for accurate win probability
        num_samples = 50000
        np.random.seed(self.seed + self.current_episode)  # Deterministic but varies by episode
        
        # Generate noise for all players
        eps1 = np.random.uniform(-self.q, self.q, num_samples)
        eps2 = np.random.uniform(-self.q, self.q, num_samples)
        eps3 = np.random.uniform(-self.q, self.q, num_samples)
        
        # Calculate total scores (effort + noise)
        score1 = e1 + eps1
        score2 = e2 + eps2
        score3 = e3 + eps3
        
        # Player 1 wins if their score is higher than both others
        wins = (score1 > score2) & (score1 > score3)
        win_probability = np.mean(wins)
        
        return win_probability
    
    def utility(self, player_id: int, effort: float, other_efforts: List[float]) -> Tuple[float, float]:
        """
        Compute expected utility for a specific player.
        
        Args:
            player_id: Player ID (0, 1, or 2)
            effort: This player's effort level
            other_efforts: List of efforts from other players
            
        Returns:
            (utility, cost): Expected utility and cost for this player
        """
        if len(other_efforts) != 2:
            raise ValueError(f"Expected 2 other efforts for 3-player game, got {len(other_efforts)}")
        
        # Reconstruct all efforts in the correct order
        all_efforts = [0.0] * 3
        all_efforts[player_id] = effort
        
        other_idx = 0
        for i in range(3):
            if i != player_id:
                all_efforts[i] = other_efforts[other_idx]
                other_idx += 1
        
        e1, e2, e3 = all_efforts
        
        # Calculate win probability for this player
        if player_id == 0:
            p_win = self.probability_win_three_players(e1, e2, e3)
        elif player_id == 1:
            p_win = self.probability_win_three_players(e2, e1, e3)
        else:  # player_id == 2
            p_win = self.probability_win_three_players(e3, e1, e2)
        
        # Calculate expected reward and cost
        expected_reward = self.w_l + p_win * (self.w_h - self.w_l)
        cost = self.k * effort * effort
        
        utility = expected_reward - cost
        
        return utility, cost
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial states for all three players
        """
        self.current_episode += 1
        
        # Return dummy states (players will learn from rewards, not states)
        state1 = torch.tensor([0.0], dtype=torch.float32)
        state2 = torch.tensor([0.0], dtype=torch.float32)
        state3 = torch.tensor([0.0], dtype=torch.float32)
        
        return state1, state2, state3
    
    def step(self, actions: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                                   torch.Tensor, 
                                                   torch.Tensor, 
                                                   bool, 
                                                   Dict[str, Any]]:
        """
        Take a step in the environment with actions from all players.
        
        Args:
            actions: Tensor of shape (3,) containing effort levels for all players
            
        Returns:
            (observations, rewards, costs, done, info)
        """
        if len(actions) != 3:
            raise ValueError(f"Expected 3 actions for 3-player game, got {len(actions)}")
        
        # Extract effort levels
        efforts = [action.item() if isinstance(action, torch.Tensor) else float(action) for action in actions]
        
        # Ensure efforts are within valid range
        low, high = self.effort_range
        efforts = [max(low, min(high, e)) for e in efforts]
        
        # Calculate utilities and costs for each player
        utilities = []
        costs = []
        win_probabilities = []
        
        for i in range(3):
            other_efforts = [efforts[j] for j in range(3) if j != i]
            utility, cost = self.utility(i, efforts[i], other_efforts)
            utilities.append(utility)
            costs.append(cost)
            
            # Calculate win probability for this player
            if i == 0:
                p_win = self.probability_win_three_players(efforts[0], efforts[1], efforts[2])
            elif i == 1:
                p_win = self.probability_win_three_players(efforts[1], efforts[0], efforts[2])
            else:  # i == 2
                p_win = self.probability_win_three_players(efforts[2], efforts[0], efforts[1])
            
            win_probabilities.append(p_win)
        
        # Store episode information
        episode_info = {
            "episode": self.current_episode,
            "efforts": efforts,
            "utilities": utilities,
            "costs": costs,
            "win_probabilities": win_probabilities,
            "winner": np.argmax(win_probabilities)
        }
        self.episode_history.append(episode_info)
        
        # Return observations (dummy states)
        observations = (
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32)
        )
        
        rewards = torch.tensor(utilities, dtype=torch.float32)
        costs_tensor = torch.tensor(costs, dtype=torch.float32)
        done = True  # Single-step environment
        
        # Build info dictionary
        info = {
            "efforts": tuple(efforts),
            "utilities": tuple(utilities),
            "costs": tuple(costs),
            "win_probabilities": tuple(win_probabilities),
            "winner": episode_info["winner"],
            "episode": self.current_episode
        }
        
        return observations, rewards, costs_tensor, done, info
    
    def get_equilibrium_analysis(self) -> Dict[str, Any]:
        """
        Analyze the current episode history to understand equilibrium behavior.
        
        Returns:
            Dictionary containing equilibrium analysis
        """
        if len(self.episode_history) < 100:
            return {"status": "insufficient_data", "message": "Need at least 100 episodes for analysis"}
        
        # Extract recent efforts
        recent_episodes = list(self.episode_history)[-100:]
        all_efforts = np.array([ep["efforts"] for ep in recent_episodes])
        
        # Calculate statistics for each player
        player_stats = []
        for i in range(3):
            player_efforts = all_efforts[:, i]
            stats = {
                "player_id": i,
                "mean_effort": float(np.mean(player_efforts)),
                "std_effort": float(np.std(player_efforts)),
                "min_effort": float(np.min(player_efforts)),
                "max_effort": float(np.max(player_efforts)),
                "median_effort": float(np.median(player_efforts))
            }
            player_stats.append(stats)
        
        # Check for convergence (effort stability)
        effort_stability = []
        for i in range(3):
            player_efforts = all_efforts[:, i]
            # Calculate coefficient of variation
            cv = np.std(player_efforts) / (np.mean(player_efforts) + 1e-8)
            effort_stability.append(cv)
        
        # Determine if players have converged to similar strategies
        mean_efforts = [stats["mean_effort"] for stats in player_stats]
        effort_variance = np.var(mean_efforts)
        
        analysis = {
            "status": "analysis_complete",
            "player_stats": player_stats,
            "effort_stability": effort_stability,
            "effort_variance": float(effort_variance),
            "converged_to_symmetric": effort_variance < 0.1,  # Threshold for symmetric equilibrium
            "total_episodes": len(self.episode_history)
        }
        
        return analysis
    
    def get_theoretical_comparison(self) -> Dict[str, Any]:
        """
        Compare current behavior with theoretical equilibrium (if available).
        
        Returns:
            Dictionary containing comparison with theoretical values
        """
        if self.e_star is None:
            return {"status": "no_theoretical", "message": "No theoretical effort provided"}
        
        analysis = self.get_equilibrium_analysis()
        if analysis["status"] != "analysis_complete":
            return analysis
        
        # Compare with theoretical equilibrium
        theoretical_effort = self.e_star
        player_efforts = [stats["mean_effort"] for stats in analysis["player_stats"]]
        
        gaps = [abs(effort - theoretical_effort) for effort in player_efforts]
        mean_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        
        comparison = {
            "theoretical_effort": theoretical_effort,
            "player_efforts": player_efforts,
            "gaps": gaps,
            "mean_gap": float(mean_gap),
            "max_gap": float(max_gap),
            "converged_to_theory": mean_gap < 0.5,  # Threshold for convergence to theory
            "analysis": analysis
        }
        
        return comparison 