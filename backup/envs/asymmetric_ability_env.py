import torch
import numpy as np

class AsymmetricAbilityEnv:
    """
    Environment for asymmetric ability parameters experiment.
    Supports different ability parameters l_i for each player while keeping costs equal.
    """
    def __init__(self, config):
        # Ability parameters for each player
        self.l_players = []
        if "l1" in config and "l2" in config:
            self.l_players = [config["l1"], config["l2"]]
        elif "l" in config:
            # Fallback to symmetric case
            self.l_players = [config["l"], config["l"]]
        else:
            raise ValueError("Config must contain either 'l1'/'l2' or 'l'")
        
        # Equal cost parameter for both players
        self.k = config["k"]
        self.q = config["q"]
        self.w_h = config["w_h"]
        self.w_l = config["w_l"]
        self.effort_range = config["effort_range"]
        self.seed = config["seed"]
        self.num_players = config.get("num_players", 2)
        
        # Store theoretical values
        self.theoretical_efforts = config.get("theoretical_efforts", [])
        self.theoretical_costs = config.get("theoretical_costs", [])
        
        if len(self.l_players) != self.num_players:
            raise ValueError(f"Number of ability parameters ({len(self.l_players)}) must match number of players ({self.num_players})")

    def get_ability_parameters(self):
        """Return ability parameters for all players"""
        return self.l_players.copy()
    
    def get_theoretical_efforts(self):
        """Return theoretical optimal efforts"""
        return self.theoretical_efforts.copy()
    
    def get_theoretical_costs(self):
        """Return theoretical optimal costs"""
        return self.theoretical_costs.copy()

    def probability_uniform_with_abilities(self, e1, e2, l1, l2):
        """
        Compute P(l1*e1 + ε1 > l2*e2 + ε2) for ε1, ε2 ~ Uniform(-q, q).
        This accounts for different ability parameters affecting effective effort.
        
        The effective efforts are l1*e1 and l2*e2.
        The difference D = (l1*e1 + ε1) - (l2*e2 + ε2) = (l1*e1 - l2*e2) + (ε1 - ε2)
        Since ε1 - ε2 has triangular distribution over [-2q, 2q], we can use the same
        closed-form CDF but with effective effort difference.
        """
        effective_diff = l2 * e2 - l1 * e1  # Note: this is e2_eff - e1_eff for the CDF
        d = torch.tensor(effective_diff, dtype=torch.float32)
        
        # Clamp d to [-2q, 2q]
        d_clamped = torch.clamp(d, -2 * self.q, 2 * self.q)
        
        # For d_clamped < 0: p = 1 - ((d_clamped + 2q)^2)/(8q^2)
        p_neg = 1.0 - (d_clamped + 2 * self.q).pow(2) / (8 * self.q * self.q)
        
        # For d_clamped >= 0: p = ((2q - d_clamped)^2)/(8q^2)
        p_pos = (2 * self.q - d_clamped).pow(2) / (8 * self.q * self.q)
        
        mask_neg = (d_clamped < 0).float()
        p_middle = mask_neg * p_neg + (1.0 - mask_neg) * p_pos
        
        # Handle extremes
        p_final = torch.where(
            d < -2 * self.q,
            torch.tensor(1.0),
            torch.where(d > 2 * self.q, torch.tensor(0.0), p_middle)
        )
        
        return p_final.item()

    def utility(self, player_id, effort, *other_efforts):
        """
        Compute expected utility for a specific player accounting for their ability.
        
        Args:
            player_id: 0 for player 1, 1 for player 2, etc.
            effort: this player's effort
            other_efforts: efforts of other players
        
        Returns:
            utility, cost: expected utility and cost for this player
        """
        if self.num_players == 2:
            if len(other_efforts) != 1:
                raise ValueError("For 2 players, need exactly 1 other effort")
            
            other_effort = other_efforts[0]
            other_player_id = 1 - player_id
            
            # Get ability parameters
            l_self = self.l_players[player_id]
            l_other = self.l_players[other_player_id]
            
            # Compute win probability with abilities
            p_win = self.probability_uniform_with_abilities(effort, other_effort, l_self, l_other)
            
        else:
            raise ValueError(f"Unsupported number of players: {self.num_players}")
        
        # Compute reward and cost
        reward = self.w_l + p_win * (self.w_h - self.w_l)
        cost = self.k * effort * effort  # Same cost parameter k for all players
        
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
        
        # Compute utility for each player
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
        
        # Build info dict with ability parameters
        info = {"efforts": tuple(efforts)}
        for i in range(self.num_players):
            info[f"p{i+1}_l"] = self.l_players[i]  # Ability parameter
            info[f"p{i+1}_cost"] = costs[i]
        
        return obs, rewards, costs_tensor, done, info

    def get_win_probabilities(self, efforts):
        """
        Compute win probabilities for all players given their efforts.
        Useful for analysis and debugging.
        """
        if len(efforts) != self.num_players:
            raise ValueError(f"Expected {self.num_players} efforts, got {len(efforts)}")
        
        if self.num_players == 2:
            e1, e2 = efforts
            l1, l2 = self.l_players
            p1_win = self.probability_uniform_with_abilities(e1, e2, l1, l2)
            p2_win = 1.0 - p1_win
            return [p1_win, p2_win]
        else:
            raise ValueError(f"Unsupported number of players: {self.num_players}")

    def analyze_equilibrium(self, efforts):
        """
        Analyze how close the given efforts are to theoretical equilibrium.
        Returns detailed analysis including gaps, utilities, and win probabilities.
        """
        if len(efforts) != self.num_players:
            raise ValueError(f"Expected {self.num_players} efforts, got {len(efforts)}")
        
        # Compute current utilities and costs
        utilities = []
        costs = []
        for i in range(self.num_players):
            other_efforts = [efforts[j] for j in range(self.num_players) if j != i]
            u, cost = self.utility(i, efforts[i], *other_efforts)
            utilities.append(u)
            costs.append(cost)
        
        # Compute win probabilities
        win_probs = self.get_win_probabilities(efforts)
        
        # Compute gaps from theoretical values
        gaps = []
        if self.theoretical_efforts:
            gaps = [abs(efforts[i] - self.theoretical_efforts[i]) for i in range(self.num_players)]
        
        analysis = {
            "efforts": efforts,
            "theoretical_efforts": self.theoretical_efforts,
            "gaps": gaps,
            "utilities": utilities,
            "costs": costs,
            "win_probabilities": win_probs,
            "ability_parameters": self.l_players,
            "cost_parameter": self.k
        }
        
        return analysis 