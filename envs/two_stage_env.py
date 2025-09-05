import torch
import numpy as np
from typing import Tuple, Dict, List, Optional

class TwoStageEnv:
    """
    Two-stage tournament environment with sequential decision making.
    
    Stage 1: Players make initial effort decisions
    Stage 2: Players make second-round decisions based on Stage 1 outcomes
    
    The final outcome is a weighted combination of both stages.
    """
    
    def __init__(self, config):
        # Basic game parameters
        self.k1 = config["k1"]  # Stage 1 cost parameter
        self.k2 = config["k2"]  # Stage 2 cost parameter
        self.q = config["q"]    # Base noise parameter
        self.w_h = config["w_h"]
        self.w_l = config["w_l"]
        self.effort_range = config["effort_range"]
        self.seed = config["seed"]
        self.num_players = config.get("num_players", 2)
        
        # Two-stage specific parameters
        self.stage1_weight = config["stage1_weight"]
        self.stage2_weight = config["stage2_weight"]
        self.stage1_noise_factor = config["stage1_noise_factor"]
        self.stage2_noise_factor = config["stage2_noise_factor"]
        # Probability model for win rate: 'logit' (recommended) or 'uniform' (legacy)
        # When 'logit', p_win = sigmoid((e_i - e_j) / (q * noise_factor))
        # This aligns with a softmax/logit victory model often used in tournaments
        self.prob_model = config.get("prob_model", "logit")
        
        # Information revelation settings
        self.information_revelation = config["information_revelation"]
        self.reveal_opponent_effort = config["reveal_opponent_effort"]
        self.reveal_stage1_outcome = config["reveal_stage1_outcome"]
        self.reveal_noise_realization = config["reveal_noise_realization"]
        
        # Theoretical values
        self.stage1_effort = config["stage1_effort"]
        self.stage2_effort = config["stage2_effort"]
        self.total_cost = config["total_cost"]
        self.expected_utility = config["expected_utility"]
        
        # Game state
        self.current_stage = 1
        self.stage1_efforts = None
        self.stage1_outcomes = None
        self.stage1_winner = None
        self.stage1_noise = None
        self.information_state = {}
        
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
    
    def get_theoretical_efforts(self) -> List[float]:
        """Return theoretical equilibrium efforts for both stages"""
        return [self.stage1_effort, self.stage2_effort]
    
    def get_stage_weights(self) -> Tuple[float, float]:
        """Return stage weights"""
        return self.stage1_weight, self.stage2_weight
    
    def probability_uniform(self, e1: float, e2: float, noise_factor: float = 1.0) -> float:
        """
        Compute P(e1 + ε1 > e2 + ε2) for ε1, ε2 ~ Uniform(-q*noise_factor, q*noise_factor).
        """
        effective_q = self.q * noise_factor
        d = torch.tensor(e2 - e1, dtype=torch.float32)
        d_clamped = torch.clamp(d, -2 * effective_q, 2 * effective_q)
        
        p_neg = 1.0 - (d_clamped + 2 * effective_q).pow(2) / (8 * effective_q * effective_q)
        p_pos = (2 * effective_q - d_clamped).pow(2) / (8 * effective_q * effective_q)
        mask_neg = (d_clamped < 0).float()
        p_middle = mask_neg * p_neg + (1.0 - mask_neg) * p_pos
        
        p_final = torch.where(
            d < -2 * effective_q,
            torch.tensor(1.0),
            torch.where(d > 2 * effective_q, torch.tensor(0.0), p_middle)
        )
        return p_final.item()

    def probability_logit(self, e1: float, e2: float, noise_factor: float = 1.0) -> float:
        """
        Logistic win probability model:
        p(win for player with effort e1 vs e2) = sigmoid((e1 - e2) / (q * noise_factor))

        Args:
            e1: Effort of player 1
            e2: Effort of player 2
            noise_factor: Stage-specific multiplicative factor on q

        Returns:
            Probability in [0,1]
        """
        effective_q = max(1e-8, float(self.q) * float(noise_factor))
        d = (float(e1) - float(e2)) / effective_q
        # Use torch.sigmoid for numerical stability; convert to Python float for consistency
        return float(torch.sigmoid(torch.tensor(d, dtype=torch.float32)).item())
    
    def compute_stage_utility(self, player_effort: float, other_efforts: List[float], 
                            stage: int) -> Tuple[float, float]:
        """
        Compute utility for a player in a specific stage.
        
        Args:
            player_effort: This player's effort
            other_efforts: List of other players' efforts
            stage: Stage number (1 or 2)
        
        Returns:
            (utility, cost) tuple
        """
        if stage == 1:
            k = self.k1
            noise_factor = self.stage1_noise_factor
        else:
            k = self.k2
            noise_factor = self.stage2_noise_factor
        
        if self.num_players == 2:
            if len(other_efforts) != 1:
                raise ValueError("For 2 players, need exactly 1 other effort")
            if self.prob_model == "logit":
                p_win = self.probability_logit(player_effort, other_efforts[0], noise_factor)
            else:
                p_win = self.probability_uniform(player_effort, other_efforts[0], noise_factor)
        else:
            raise ValueError(f"Unsupported number of players: {self.num_players}")
        
        reward = self.w_l + p_win * (self.w_h - self.w_l)
        cost = k * player_effort * player_effort
        return reward - cost, cost
    
    def simulate_stage_outcome(self, efforts: List[float], stage: int) -> Dict:
        """
        Simulate the actual outcome of a stage with noise.
        
        Args:
            efforts: List of efforts for all players
            stage: Stage number (1 or 2)
        
        Returns:
            Dictionary with outcome information
        """
        if stage == 1:
            noise_factor = self.stage1_noise_factor
        else:
            noise_factor = self.stage2_noise_factor
        
        effective_q = self.q * noise_factor
        
        if self.prob_model == "logit" and self.num_players == 2:
            # Sample winner from Bernoulli with logistic probability
            p_win_0 = self.probability_logit(efforts[0], efforts[1], noise_factor)
            u = np.random.rand()
            winner = 0 if u < p_win_0 else 1
            # Synthesize noise values for record-keeping (not used in logic)
            noise_values = [0.0, 0.0]
            total_efforts = [efforts[0], efforts[1]]
        else:
            # Legacy uniform noise model
            noise_values = []
            for i in range(self.num_players):
                noise = np.random.uniform(-effective_q, effective_q)
                noise_values.append(noise)
            total_efforts = [efforts[i] + noise_values[i] for i in range(self.num_players)]
            winner = np.argmax(total_efforts)
        
        return {
            "efforts": efforts,
            "noise_values": noise_values,
            "total_efforts": total_efforts,
            "winner": winner,
            "stage": stage
        }
    
    def get_information_state(self, player_id: int) -> Dict:
        """
        Get the information available to a specific player before Stage 2.
        
        Args:
            player_id: Player ID (0-indexed)
        
        Returns:
            Dictionary with available information
        """
        info = {"stage": 2, "player_id": player_id}
        
        if self.information_revelation == "none":
            # No information revealed
            pass
        elif self.information_revelation == "partial":
            # Reveal some information based on settings
            if self.reveal_stage1_outcome and self.stage1_outcomes:
                info["stage1_winner"] = self.stage1_outcomes["winner"]
                info["won_stage1"] = (self.stage1_outcomes["winner"] == player_id)
            
            if self.reveal_opponent_effort and self.stage1_efforts:
                info["opponent_stage1_efforts"] = [
                    self.stage1_efforts[i] for i in range(self.num_players) if i != player_id
                ]
        elif self.information_revelation == "full":
            # Reveal all information
            if self.stage1_efforts:
                info["stage1_efforts"] = self.stage1_efforts
            if self.stage1_outcomes:
                info["stage1_outcomes"] = self.stage1_outcomes
        
        return info
    
    def reset(self) -> Tuple[torch.Tensor, ...]:
        """
        Reset the environment to Stage 1.
        
        Returns:
            Initial states for all players
        """
        self.current_stage = 1
        self.stage1_efforts = None
        self.stage1_outcomes = None
        self.stage1_winner = None
        self.stage1_noise = None
        self.information_state = {}
        
        # Return initial states as a fixed-length informative vector placeholder for Stage 1
        # state = [stage_indicator(=1.0), won_stage1(=0.0), my_e1(=0.0), opp_e1(=0.0), p_win_estimate(=0.0)]
        return tuple(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(self.num_players))
    
    def step_stage1(self, actions: List[torch.Tensor]) -> Tuple:
        """
        Execute Stage 1 with given actions.
        
        Args:
            actions: List of effort tensors for each player
        
        Returns:
            (next_states, rewards, costs, done, info)
        """
        if len(actions) != self.num_players:
            raise ValueError(f"Expected {self.num_players} actions, got {len(actions)}")
        
        # Extract efforts
        efforts = [action.item() for action in actions]
        self.stage1_efforts = efforts
        
        # Simulate Stage 1 outcome
        self.stage1_outcomes = self.simulate_stage_outcome(efforts, stage=1)
        self.stage1_winner = self.stage1_outcomes["winner"]
        
        # Compute Stage 1 utilities and costs
        utilities = []
        costs = []
        
        for i in range(self.num_players):
            other_efforts = [efforts[j] for j in range(self.num_players) if j != i]
            u, cost = self.compute_stage_utility(efforts[i], other_efforts, stage=1)
            # Weight Stage-1 utility at the source to avoid double counting later
            utilities.append(self.stage1_weight * u)
            costs.append(cost)
        
        # Prepare states for Stage 2 (include information)
        next_states = []
        for i in range(self.num_players):
            # Encode an informative Stage-2 state vector depending on revelation settings
            # state = [stage_indicator(=2.0), won_stage1(0/1 or 0), my_e1, opp_e1(0 if hidden), p_win_estimate(optional or 0)]
            won_flag = 0.0
            my_e1 = float(efforts[i])
            opp_e1 = 0.0
            p_win_est = 0.0

            if self.information_revelation == "full":
                if self.stage1_outcomes is not None:
                    won_flag = 1.0 if self.stage1_outcomes["winner"] == i else 0.0
                # In full mode, expose both efforts
                other_idx = 1 - i if self.num_players == 2 else None
                if other_idx is not None and self.stage1_efforts is not None:
                    opp_e1 = float(self.stage1_efforts[other_idx])
                # Optional: estimate win probability for Stage 2 based on Stage 1 signal
                if other_idx is not None and self.stage1_efforts is not None:
                    if self.prob_model == "logit":
                        p_win_est = self.probability_logit(my_e1, opp_e1, self.stage1_noise_factor)
                    else:
                        p_win_est = self.probability_uniform(my_e1, opp_e1, self.stage1_noise_factor)
            elif self.information_revelation == "partial":
                # Reveal outcome if configured
                if self.reveal_stage1_outcome and self.stage1_outcomes is not None:
                    won_flag = 1.0 if self.stage1_outcomes["winner"] == i else 0.0
                # Reveal opponent effort if configured
                if self.reveal_opponent_effort and self.stage1_efforts is not None:
                    other_idx = 1 - i if self.num_players == 2 else None
                    if other_idx is not None:
                        opp_e1 = float(self.stage1_efforts[other_idx])
                # Optional probability if both efforts visible
                if opp_e1 != 0.0:
                    if self.prob_model == "logit":
                        p_win_est = self.probability_logit(my_e1, opp_e1, self.stage1_noise_factor)
                    else:
                        p_win_est = self.probability_uniform(my_e1, opp_e1, self.stage1_noise_factor)
            else:
                # none: expose only the stage indicator and own effort
                pass

            next_states.append(torch.tensor([2.0, won_flag, my_e1, opp_e1, float(p_win_est)]))
        
        # Update stage
        self.current_stage = 2
        
        # Build info dict
        info = {
            "stage": 1,
            "efforts": tuple(efforts),
            "stage1_winner": self.stage1_winner,
            "stage1_outcomes": self.stage1_outcomes
        }
        
        for i in range(self.num_players):
            info[f"p{i+1}_stage1_cost"] = costs[i]
            info[f"p{i+1}_information"] = self.get_information_state(i)
        
        return (
            tuple(next_states),
            torch.tensor(utilities, dtype=torch.float32),
            torch.tensor(costs, dtype=torch.float32),
            False,  # Not done yet, Stage 2 remains
            info
        )
    
    def step_stage2(self, actions: List[torch.Tensor]) -> Tuple:
        """
        Execute Stage 2 with given actions and compute final outcomes.
        
        Args:
            actions: List of effort tensors for each player
        
        Returns:
            (final_states, rewards_stage2_weighted, total_costs, done, info)
        """
        if len(actions) != self.num_players:
            raise ValueError(f"Expected {self.num_players} actions, got {len(actions)}")
        
        if self.stage1_efforts is None:
            raise ValueError("Stage 1 must be completed before Stage 2")
        
        # Extract Stage 2 efforts
        stage2_efforts = [action.item() for action in actions]
        
        # Simulate Stage 2 outcome
        stage2_outcomes = self.simulate_stage_outcome(stage2_efforts, stage=2)
        
        # Compute Stage 2 utilities and costs
        stage2_utilities = []
        stage2_costs = []
        
        for i in range(self.num_players):
            other_efforts = [stage2_efforts[j] for j in range(self.num_players) if j != i]
            u, cost = self.compute_stage_utility(stage2_efforts[i], other_efforts, stage=2)
            stage2_utilities.append(u)
            stage2_costs.append(cost)
        
        # Compute Stage-1 raw utilities (for logging) and costs; return only Stage-2 weighted utility as reward
        stage1_utilities = []
        stage1_costs = []
        total_costs = []
        for i in range(self.num_players):
            # Recompute Stage-1 cost for completeness
            other_efforts = [self.stage1_efforts[j] for j in range(self.num_players) if j != i]
            u1, cost1 = self.compute_stage_utility(self.stage1_efforts[i], other_efforts, stage=1)
            stage1_utilities.append(u1)
            stage1_costs.append(cost1)
            total_costs.append(cost1 + stage2_costs[i])
        
        # Weight Stage-2 utilities only
        rewards_stage2_weighted = [self.stage2_weight * u2 for u2 in stage2_utilities]
        
        # Final states as zeroed informative vectors matching state shape
        final_states = tuple(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(self.num_players))
        
        # Build comprehensive info dict
        info = {
            "stage": 2,
            "stage1_efforts": tuple(self.stage1_efforts),
            "stage2_efforts": tuple(stage2_efforts),
            "stage1_outcomes": self.stage1_outcomes,
            "stage2_outcomes": stage2_outcomes,
            "stage1_winner": self.stage1_outcomes["winner"],
            "stage2_winner": stage2_outcomes["winner"],
            "stage_weights": (self.stage1_weight, self.stage2_weight)
        }
        
        for i in range(self.num_players):
            info[f"p{i+1}_stage1_cost"] = stage1_costs[i]
            info[f"p{i+1}_stage2_cost"] = stage2_costs[i]
            info[f"p{i+1}_total_cost"] = total_costs[i]
            info[f"p{i+1}_stage1_utility"] = stage1_utilities[i]
            info[f"p{i+1}_stage2_utility"] = stage2_utilities[i]
        
        return (
            final_states,
            torch.tensor(rewards_stage2_weighted, dtype=torch.float32),
            torch.tensor(total_costs, dtype=torch.float32),
            True,  # Game is done
            info
        )
    
    def step(self, actions: List[torch.Tensor]) -> Tuple:
        """
        Execute one step of the environment.
        
        Args:
            actions: List of effort tensors for each player
        
        Returns:
            (states, rewards, costs, done, info)
        """
        if self.current_stage == 1:
            return self.step_stage1(actions)
        elif self.current_stage == 2:
            return self.step_stage2(actions)
        else:
            raise ValueError(f"Invalid stage: {self.current_stage}")
    
    def get_current_stage(self) -> int:
        """Get the current stage number"""
        return self.current_stage
    
    def is_stage1_complete(self) -> bool:
        """Check if Stage 1 has been completed"""
        return self.stage1_efforts is not None
    
    def get_stage1_information(self, player_id: int) -> Dict:
        """Get Stage 1 information available to a specific player"""
        return self.get_information_state(player_id)
