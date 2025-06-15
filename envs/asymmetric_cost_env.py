import torch
import numpy as np

class AsymmetricCostEnv:
    """
    Environment for asymmetric cost parameters experiment.
    Supports different cost parameters k_i for each player.
    """
    def __init__(self, config):
        # Cost parameters for each player
        self.k_players = []
        if "k1" in config and "k2" in config:
            self.k_players = [config["k1"], config["k2"]]
        elif "k" in config:
            # Fallback to symmetric case
            self.k_players = [config["k"], config["k"]]
        else:
            raise ValueError("Config must contain either 'k1'/'k2' or 'k'")
        
        self.q = config["q"]
        self.w_h = config["w_h"]
        self.w_l = config["w_l"]
        self.effort_range = config["effort_range"]
        self.seed = config["seed"]
        self.num_players = config.get("num_players", 2)
        
        # Store theoretical values if available
        self.theoretical_efforts = config.get("theoretical_efforts", None)
        self.theoretical_costs = config.get("theoretical_costs", None)
        
        # For backward compatibility with symmetric case
        if len(self.k_players) == 1:
            self.k_players = self.k_players * self.num_players

    def probability_uniform(self, e1, e2):
        """
        Compute P(e1 + ε1 > e2 + ε2) for ε1, ε2 ~ Uniform(-q, q).
        The difference D = ε1 - ε2 has a triangular PDF over [-2q, 2q].
        Closed-form CDF for P(D > e2 - e1).
        """
        d = torch.tensor(e2 - e1, dtype=torch.float32)
        # Clamp d to [-2q, 2q]
        d_clamped = torch.clamp(d, -2 * self.q, 2 * self.q)
        # For d_clamped < 0: p = 1 - ((d_clamped + 2q)^2)/(8q^2)
        p_neg = 1.0 - (d_clamped + 2 * self.q).pow(2) / (8 * self.q * self.q)
        # For d_clamped >= 0: p = ((2q - d_clamped)^2)/(8q^2)
        p_pos = (2 * self.q - d_clamped).pow(2) / (8 * self.q * self.q)
        mask_neg = (d_clamped < 0).float()
        p_middle = mask_neg * p_neg + (1.0 - mask_neg) * p_pos
        # Handle extremes: if d < -2q => probability = 1; if d > 2q => probability = 0
        p_final = torch.where(
            d < -2 * self.q,
            torch.tensor(1.0),
            torch.where(d > 2 * self.q, torch.tensor(0.0), p_middle)
        )
        return p_final.item()

    def probability_win_three_players_analytical(self, e1, e2, e3):
        """
        Analytical calculation for 3-player win probability.
        For asymmetric costs, this becomes more complex.
        """
        if abs(e2 - e3) < 1e-6:  # Symmetric case for players 2 and 3
            e_diff = e1 - e2
            # Use linear approximation around equilibrium
            marginal_effect = 1.0 / (2.0 * self.q)
            prob = 1.0/3.0 + marginal_effect * e_diff
            prob = max(0.0, min(1.0, prob))
            return prob
        else:
            # For asymmetric case, use Monte Carlo
            num_samples = 20000
            np.random.seed(42)
            
            eps1 = np.random.uniform(-self.q, self.q, num_samples)
            eps2 = np.random.uniform(-self.q, self.q, num_samples)
            eps3 = np.random.uniform(-self.q, self.q, num_samples)
            
            effort1_total = e1 + eps1
            effort2_total = e2 + eps2
            effort3_total = e3 + eps3
            
            wins = (effort1_total > effort2_total) & (effort1_total > effort3_total)
            return np.mean(wins)

    def utility(self, player_id, effort, *other_efforts):
        """
        Compute expected utility for a specific player with their cost parameter.
        
        Args:
            player_id: Index of the player (0-based)
            effort: Effort level of this player
            other_efforts: Effort levels of other players
        """
        if player_id >= len(self.k_players):
            raise ValueError(f"Player ID {player_id} exceeds number of cost parameters")
        
        # Get this player's cost parameter
        k_player = self.k_players[player_id]
        
        if self.num_players == 2:
            if len(other_efforts) != 1:
                raise ValueError("For 2 players, need exactly 1 other effort")
            e_other = other_efforts[0]
            p_win = self.probability_uniform(effort, e_other)
        elif self.num_players == 3:
            if len(other_efforts) != 2:
                raise ValueError("For 3 players, need exactly 2 other efforts")
            e2, e3 = other_efforts
            p_win = self.probability_win_three_players_analytical(effort, e2, e3)
        else:
            raise ValueError(f"Unsupported number of players: {self.num_players}")
        
        reward = self.w_l + p_win * (self.w_h - self.w_l)
        cost = k_player * effort * effort  # Use player-specific cost parameter
        return reward - cost, cost

    def reset(self):
        """Return initial states for all players"""
        return tuple(torch.tensor([0.0]) for _ in range(self.num_players))

    def step(self, actions):
        """
        Takes the effort of all players and returns reward and cost.
        Input: actions is a tensor of length num_players
        Returns: obs, rewards, costs, done, info
        """
        if len(actions) != self.num_players:
            raise ValueError(f"Expected {self.num_players} actions, got {len(actions)}")
        
        efforts = [action.item() for action in actions]
        utilities = []
        costs = []
        
        # Compute utility for each player using their specific cost parameter
        for i in range(self.num_players):
            other_efforts = [efforts[j] for j in range(self.num_players) if j != i]
            u, cost = self.utility(i, efforts[i], *other_efforts)
            utilities.append(u)
            costs.append(cost)
        
        # Return observations (dummy states for all players)
        obs = tuple(torch.tensor([0.0]) for _ in range(self.num_players))
        rewards = torch.tensor(utilities, dtype=torch.float32)
        costs_tensor = torch.tensor(costs, dtype=torch.float32)
        done = True
        
        # Build info dict
        info = {"efforts": tuple(efforts)}
        for i in range(self.num_players):
            info[f"p{i+1}_cost"] = costs[i]
            info[f"p{i+1}_k"] = self.k_players[i]
        
        return obs, rewards, costs_tensor, done, info

    def get_theoretical_efforts(self):
        """Return theoretical equilibrium efforts for each player"""
        return self.theoretical_efforts if self.theoretical_efforts else None

    def get_theoretical_costs(self):
        """Return theoretical equilibrium costs for each player"""
        return self.theoretical_costs if self.theoretical_costs else None

    def get_cost_parameters(self):
        """Return cost parameters for each player"""
        return self.k_players.copy() 