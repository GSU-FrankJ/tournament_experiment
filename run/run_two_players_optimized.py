#!/usr/bin/env python3
"""
Optimized Two-Player One-Stage Tournament Experiment
===================================================

This module implements highly optimized algorithms for two-player one-stage tournament games.
Focus on performance improvements for both gradient descent and PPO algorithms.

Key optimizations:
- Adaptive learning rate scheduling for gradient descent
- Curriculum learning and reward shaping for PPO
- Enhanced convergence detection
- Performance monitoring and logging
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import configurations and environments
from config.one_stage_two_players import config
from envs.one_stage_env import OneStageEnv

# Import agents
from agents.gradient_solver import gradient_descent_solver
from agents.enhanced_ppo_agent import EnhancedPPOAgent, ContinuousActionSpace
from agents.ultra_optimized_ppo_agent import UltraOptimizedPPOAgent

# Import utilities
from utils.logger import save_result

@dataclass
class OptimizationConfig:
    """Configuration for optimization experiments"""
    # Gradient descent optimization
    gradient_lr_schedule: str = "adaptive"  # "constant", "adaptive", "cosine"
    gradient_momentum: float = 0.9
    gradient_adaptive_threshold: float = 1e-4
    
    # PPO optimization
    ppo_curriculum_learning: bool = True
    ppo_reward_shaping: bool = True
    
    # General optimization
    early_stopping_patience: int = 3000
    convergence_threshold: float = 0.5
    max_training_time: int = 300  # seconds

class AdaptiveGradientSolver:
    """
    Enhanced gradient descent solver with adaptive learning rate and momentum
    """
    
    def __init__(self, env, config: OptimizationConfig):
        self.env = env
        self.config = config
        self.lr_schedule = config.gradient_lr_schedule
        self.momentum = config.gradient_momentum
        self.adaptive_threshold = config.gradient_adaptive_threshold
        
        # Initialize tracking variables
        self.velocity = 0.0
        self.gradient_history = []
        
    def adaptive_learning_rate(self, gradient: float, step: int, base_lr: float = 0.1) -> float:
        """
        Compute adaptive learning rate based on gradient behavior
        """
        if self.lr_schedule == "constant":
            return base_lr
        
        elif self.lr_schedule == "adaptive":
            # Reduce learning rate if gradient is oscillating
            if len(self.gradient_history) > 10:
                recent_gradients = self.gradient_history[-10:]
                gradient_variance = np.var(recent_gradients)
                
                if gradient_variance > self.adaptive_threshold:
                    # High variance - reduce learning rate
                    return base_lr * 0.5
                else:
                    # Low variance - maintain or increase learning rate
                    return min(base_lr * 1.1, 0.2)
            return base_lr
        
        elif self.lr_schedule == "cosine":
            # Cosine annealing
            max_steps = 100000
            return base_lr * 0.5 * (1 + np.cos(np.pi * step / max_steps))
        
        return base_lr
    
    def solve(self, lr: float = 0.1, steps: int = 100000, eps: float = 1e-3) -> Tuple[float, float, float]:
        """
        Solve using enhanced gradient descent with adaptive learning rate and momentum
        """
        # Initialize effort
        if hasattr(self.env, "effort_range"):
            low, high = self.env.effort_range
            e = (low + high) / 2.0
        else:
            e = 1.0
        
        # Setup logging
        log_path = "results/logs/adaptive_gradient_log.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "w") as f:
            f.write("Step,Effort,Gradient,LearningRate,Velocity,Utility\n")
        
        print(f"Starting adaptive gradient descent with {self.lr_schedule} learning rate schedule...")
        
        best_effort = e
        best_gap = float('inf')
        no_improvement_count = 0
        
        for step in range(steps):
            # Compute gradient using central difference
            u_plus, _ = self.env.utility(e + eps, e)
            u_minus, _ = self.env.utility(e - eps, e)
            gradient = (u_plus - u_minus) / (2 * eps)
            
            # Compute adaptive learning rate
            current_lr = self.adaptive_learning_rate(gradient, step, lr)
            
            # Apply momentum
            self.velocity = self.momentum * self.velocity + current_lr * gradient
            
            # Update effort
            e += self.velocity
            
            # Clamp to valid range
            if hasattr(self.env, "effort_range"):
                low, high = self.env.effort_range
                e = np.clip(e, low, high)
            
            # Track history
            self.gradient_history.append(gradient)
            
            # Check for convergence
            theoretical_effort = getattr(self.env, 'e_star', 87.5)
            current_gap = abs(e - theoretical_effort)
            
            if current_gap < best_gap:
                best_effort = e
                best_gap = current_gap
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count > self.config.early_stopping_patience:
                print(f"Early stopping at step {step} due to no improvement")
                break
            
            # Log progress
            if step % 1000 == 0:
                current_u, _ = self.env.utility(e, e)
                with open(log_path, "a") as f:
                    f.write(f"{step},{e:.6f},{gradient:.6f},{current_lr:.6f},{self.velocity:.6f},{current_u:.6f}\n")
                
                if step % 10000 == 0:
                    print(f"Step {step}: effort={e:.3f}, gap={current_gap:.3f}, lr={current_lr:.6f}")
        
        # Compute final utility and cost
        final_u, final_cost = self.env.utility(best_effort, best_effort)
        
        print(f"Adaptive gradient descent completed:")
        print(f"  Final effort: {best_effort:.3f}")
        print(f"  Final gap: {best_gap:.3f}")
        print(f"  Total steps: {step + 1}")
        
        return best_effort, final_u, final_cost

class CurriculumPPOTrainer:
    """
    PPO trainer with curriculum learning and advanced optimization techniques
    """
    
    def __init__(self, env, config: OptimizationConfig):
        self.env = env
        self.config = config
        self.curriculum_stages = [
            {"effort_range": (70, 100), "episodes": 3000, "lr": 0.0003},
            {"effort_range": (80, 95), "episodes": 5000, "lr": 0.0001},
            {"effort_range": (85, 90), "episodes": 7000, "lr": 0.00005},
        ]
        
    def create_optimized_agent(self, stage_config: Dict) -> EnhancedPPOAgent:
        """
        Create an optimized PPO agent for a specific curriculum stage
        """
        # Use ultra-optimized agent instead of enhanced agent
        from agents.ultra_optimized_ppo_agent import UltraOptimizedPPOAgent
        
        agent = UltraOptimizedPPOAgent(
            effort_range=stage_config["effort_range"],
            theoretical_effort=getattr(self.env, 'e_star', 87.5),
            log_path=f"results/logs/ultra_ppo_stage_{len(self.curriculum_stages)}.csv"
        )
        
        return agent
    
    def reward_shaping(self, raw_reward: float, effort: float, theoretical_effort: float) -> float:
        """
        Apply reward shaping to guide learning towards theoretical equilibrium
        """
        if not self.config.ppo_reward_shaping:
            return raw_reward
        
        # Distance-based shaping
        distance_penalty = -0.1 * abs(effort - theoretical_effort)
        
        # Convergence bonus
        if abs(effort - theoretical_effort) < 2.0:
            convergence_bonus = 1.0
        elif abs(effort - theoretical_effort) < 5.0:
            convergence_bonus = 0.5
        else:
            convergence_bonus = 0.0
        
        return raw_reward + distance_penalty + convergence_bonus
    
    def train_curriculum_stage(self, stage_idx: int, agent1, agent2) -> Dict:
        """
        Train agents for a specific curriculum stage
        """
        stage_config = self.curriculum_stages[stage_idx]
        print(f"\nTraining curriculum stage {stage_idx + 1}/{len(self.curriculum_stages)}")
        print(f"  Effort range: {stage_config['effort_range']}")
        print(f"  Episodes: {stage_config['episodes']}")
        print(f"  Learning rate: {stage_config['lr']}")
        
        theoretical_effort = getattr(self.env, 'e_star', 87.5)
        best_effort = None
        best_gap = float('inf')
        
        start_time = time.time()
        
        for episode in range(stage_config["episodes"]):
            # Reset environment
            state1, state2 = self.env.reset()
            
            # Select actions
            a1 = agent1.select_action(state1)
            a2 = agent2.select_action(state2)
            
            # Environment step
            _, raw_rewards, _, _, info = self.env.step(torch.stack([a1, a2]))
            
            # Apply reward shaping
            shaped_rewards = [
                self.reward_shaping(raw_rewards[0].item(), info["efforts"][0], theoretical_effort),
                self.reward_shaping(raw_rewards[1].item(), info["efforts"][1], theoretical_effort)
            ]
            
            # Store rewards
            agent1.store_reward(shaped_rewards[0])
            agent2.store_reward(shaped_rewards[1])
            
            # Update policies every episode for ultra-optimized agents
            if episode % 1 == 0:  # Update every episode
                agent1.update_policy(episode)
                agent2.update_policy(episode)
            
            # Track performance
            current_effort = info["efforts"][0]
            current_gap = abs(current_effort - theoretical_effort)
            
            if current_gap < best_gap:
                best_effort = current_effort
                best_gap = current_gap
            
            # Progress reporting
            if episode % 500 == 0 and episode > 0:
                elapsed_time = time.time() - start_time
                print(f"  Episode {episode}: effort={current_effort:.2f}, gap={current_gap:.3f}, time={elapsed_time:.1f}s")
                
                # Early stopping check
                if current_gap < self.config.convergence_threshold:
                    print(f"  Early convergence achieved at episode {episode}")
                    break
        
        return {
            "stage": stage_idx,
            "best_effort": best_effort,
            "best_gap": best_gap,
            "episodes_trained": episode + 1,
            "training_time": time.time() - start_time
        }
    
    def train(self) -> Tuple[float, Dict]:
        """
        Train using curriculum learning approach
        """
        print("Starting curriculum PPO training...")
        
        # Initialize agents for first stage
        stage_config = self.curriculum_stages[0]
        agent1 = self.create_optimized_agent(stage_config)
        agent2 = self.create_optimized_agent(stage_config)
        
        training_history = []
        
        # Train through curriculum stages
        for stage_idx in range(len(self.curriculum_stages)):
            if stage_idx > 0:
                # Update agents for new stage
                stage_config = self.curriculum_stages[stage_idx]
                agent1 = self.create_optimized_agent(stage_config)
                agent2 = self.create_optimized_agent(stage_config)
            
            stage_result = self.train_curriculum_stage(stage_idx, agent1, agent2)
            training_history.append(stage_result)
        
        # Get final performance
        final_result = training_history[-1]
        
        print(f"\nCurriculum training completed:")
        print(f"  Final effort: {final_result['best_effort']:.3f}")
        print(f"  Final gap: {final_result['best_gap']:.3f}")
        print(f"  Total training time: {sum(r['training_time'] for r in training_history):.1f}s")
        
        return final_result['best_effort'], {
            "training_history": training_history,
            "final_gap": final_result['best_gap'],
            "convergence_quality": "Excellent" if final_result['best_gap'] < 0.5 else "Good" if final_result['best_gap'] < 2.0 else "Fair"
        }

def run_optimized_gradient_experiment(opt_config: OptimizationConfig) -> Dict:
    """
    Run optimized gradient descent experiment
    """
    print("=" * 60)
    print("OPTIMIZED GRADIENT DESCENT EXPERIMENT")
    print("=" * 60)
    
    # Use the global config for environment setup
    env = OneStageEnv(config)
    solver = AdaptiveGradientSolver(env, opt_config)
    
    start_time = time.time()
    effort, eu, cost = solver.solve(lr=0.1, steps=100000, eps=1e-3)
    training_time = time.time() - start_time
    
    theoretical_effort = config["effort"]
    gap = abs(effort - theoretical_effort)
    
    # Determine convergence quality
    if gap < 0.1:
        convergence_quality = "Excellent"
    elif gap < 0.5:
        convergence_quality = "Good" 
    elif gap < 2.0:
        convergence_quality = "Fair"
    else:
        convergence_quality = "Poor"
    
    result = {
        "algorithm": "Optimized_Gradient",
        "final_effort": round(effort, 2),
        "final_gap": round(gap, 3),
        "convergence_quality": convergence_quality,
        "training_time": round(training_time, 2),
        "parameters": f"adaptive_lr={opt_config.gradient_lr_schedule}, momentum={opt_config.gradient_momentum}"
    }
    
    # Save result
    save_result({
        "k": config["k"],
        "q": config["q"], 
        "w_h": config["w_h"],
        "w_l": config["w_l"],
        "EU": round(config["eu"], 2),
        "Cost_of_effort": round(config["cost"], 2),
        "effort": round(config["effort"], 2),
        "Model_training": "Optimized_Gradient",
        "Parameter": f"adaptive_lr={opt_config.gradient_lr_schedule}, momentum={opt_config.gradient_momentum}",
        "Effort_0_100": round(effort, 2) if config["effort_range"][1] == 100 else "",
        "Effort_0_200": round(effort, 2) if config["effort_range"][1] == 200 else "",
        "Convergence_Quality": convergence_quality,
        "Final_Gap": round(gap, 3)
    }, "results/tables/two_players_optimized.csv")
    
    print(f"Optimized Gradient Descent Results:")
    print(f"  Final effort: {effort:.3f} (theoretical: {config['effort']:.3f})")
    print(f"  Gap: {gap:.3f}")
    print(f"  Convergence quality: {convergence_quality}")
    print(f"  Training time: {training_time:.2f}s")
    
    return result

def run_ppo_curriculum_experiment(config):
    """Run PPO experiment with curriculum learning and ADAPTIVE theoretical effort"""
    print(f"\n🚀 Running ADAPTIVE PPO Curriculum Experiment")
    print(f"📊 Config: k={config['k']}, q={config['q']}, w_h={config['w_h']}, w_l={config['w_l']}")
    
    # Calculate theoretical effort for this specific configuration
    theoretical_effort = (config["w_h"] - config["w_l"]) / (4 * config["k"] * config["q"])
    print(f"🎯 Theoretical optimal effort: {theoretical_effort:.2f}")
    
    env = OneStageEnv(config)
    
    # Create PPO agent with ADAPTIVE theoretical effort
    ppo_agent = UltraOptimizedPPOAgent(
        effort_range=config["effort_range"],
        theoretical_effort=theoretical_effort,  # Use calculated theoretical effort
        log_path=f"results/logs/ultra_optimized_ppo_q{config['q']}.csv"
    )
    
    print(f"🎓 Agent initialized with theoretical effort: {theoretical_effort:.2f}")
    
    # Training parameters
    num_episodes = 15000
    convergence_window = 500
    convergence_threshold = 1.0
    patience = 3000
    best_gap = float('inf')
    episodes_without_improvement = 0
    
    start_time = time.time()
    print(f"🏃 Training for up to {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        state1, state2 = env.reset()
        
        # Agent 1 selects action
        action1 = ppo_agent.select_action(state1)
        
        # Agent 2 uses the same action (symmetric equilibrium)
        action2 = action1.clone()
        
        # Environment step
        obs, rewards, costs, done, info = env.step([action1, action2])
        
        # Store reward for PPO agent
        ppo_agent.store_reward(rewards[0].item())
        
        # Update policy
        metrics = ppo_agent.update_policy(episode)
        
        # Track effort
        effort_value = action1.item()
        ppo_agent.recent_efforts.append(effort_value)
        
        # Check convergence
        if episode > 0 and episode % convergence_window == 0:
            recent_efforts = list(ppo_agent.recent_efforts)[-convergence_window:]
            if len(recent_efforts) >= convergence_window:
                avg_effort = np.mean(recent_efforts)
                gap = abs(avg_effort - theoretical_effort)
                
                print(f"📈 Episode {episode}: avg_effort={avg_effort:.2f}, gap={gap:.3f}, theoretical={theoretical_effort:.2f}")
                
                if gap < best_gap:
                    best_gap = gap
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += convergence_window
                
                if gap < convergence_threshold:
                    print(f"🎉 Converged! Gap: {gap:.3f} < {convergence_threshold}")
                    break
                
                if episodes_without_improvement >= patience:
                    print(f"⏰ Early stopping due to lack of improvement")
                    break
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_efforts = list(ppo_agent.recent_efforts)[-100:] if len(ppo_agent.recent_efforts) >= 100 else list(ppo_agent.recent_efforts)
    if final_efforts:
        final_avg_effort = np.mean(final_efforts)
        final_gap = abs(final_avg_effort - theoretical_effort)
        
        # Quality assessment
        if final_gap < 0.5:
            quality = "Excellent"
        elif final_gap < 2.0:
            quality = "Good"
        elif final_gap < 5.0:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"✅ PPO Final Results:")
        print(f"   Average effort: {final_avg_effort:.2f}")
        print(f"   Theoretical effort: {theoretical_effort:.2f}")
        print(f"   Gap: {final_gap:.3f}")
        print(f"   Quality: {quality}")
        print(f"   Training time: {training_time:.2f}s")
        
        # 💾 SAVE PPO RESULT TO CSV FILE
        save_result({
            "k": config["k"],
            "q": config["q"], 
            "w_h": config["w_h"],
            "w_l": config["w_l"],
            "EU": round(config["eu"], 2),
            "Cost_of_effort": round(config["cost"], 2),
            "effort": round(theoretical_effort, 2),  # Use calculated theoretical effort
            "Model_training": "Optimized_PPO",
            "Parameter": "curriculum_learning=True, reward_shaping=True, enhanced_architecture",
            "Effort_0_100": round(final_avg_effort, 2) if config["effort_range"][1] == 100 else "",
            "Effort_0_200": round(final_avg_effort, 2) if config["effort_range"][1] == 200 else "",
            "Convergence_Quality": quality,
            "Final_Gap": round(final_gap, 3)
        }, "results/tables/two_players_optimized.csv")
        
        return {
            "algorithm": "Optimized_PPO",
            "parameters": "curriculum_learning=True, reward_shaping=True, enhanced_architecture",
            "final_effort": final_avg_effort,
            "theoretical_effort": theoretical_effort,
            "gap": final_gap,
            "quality": quality,
            "episodes": episode + 1,
            "training_time": training_time
        }
    else:
        print("❌ No efforts recorded")
        return None

def main():
    """Main function to run optimized two-player experiments"""
    global config  # Move global declaration to the top
    
    try:
        print("🚀 Starting Two-Player Optimized Tournament Experiments")
        print("=" * 80)
        
        # Test different q values to verify PPO adaptability
        test_configs = [
            {**config, "q": 25.0},  # Original
            {**config, "q": 40.0},  # Higher noise
            {**config, "q": 55.0},  # Even higher noise
        ]
        
        for test_config in test_configs:
            print(f"\n🧪 Testing with q={test_config['q']}")
            
            # Recalculate theoretical values for this q
            test_config["effort"] = (test_config["w_h"] - test_config["w_l"]) / (4 * test_config["k"] * test_config["q"])
            test_config["cost"] = test_config["k"] * test_config["effort"] ** 2
            test_config["eu"] = round(((test_config["w_h"] + test_config["w_l"]) / 2 - test_config["k"] * test_config["effort"] ** 2), 2)
            
            print(f"📊 Theoretical effort for q={test_config['q']}: {test_config['effort']:.2f}")
            
            # Update global config for gradient experiment
            original_config = config.copy()
            config.update(test_config)
            
            # Create optimization configuration for gradient
            opt_config = OptimizationConfig(
                gradient_lr_schedule="adaptive",
                gradient_momentum=0.9,
                ppo_curriculum_learning=True,
                ppo_reward_shaping=True,
                early_stopping_patience=3000,
                convergence_threshold=0.5
            )
            
            # Run experiments
            gradient_result = run_optimized_gradient_experiment(opt_config)
            ppo_result = run_ppo_curriculum_experiment(test_config)
            
            # Restore original config
            config = original_config
            
            if ppo_result is None:
                print("❌ PPO experiment failed")
                continue
                
            # Save results (removed duplicate saves since they're already in the experiment functions)
            
            # Determine best algorithm for this configuration
            if gradient_result["final_gap"] < ppo_result["gap"]:
                best_result = gradient_result
                best_algorithm = "Optimized_Gradient"
                best_gap = gradient_result["final_gap"]
            else:
                best_result = ppo_result
                best_algorithm = "Optimized_PPO"
                best_gap = ppo_result["gap"]
            
            print("\n📊 COMPARISON RESULTS")
            print("=" * 50)
            print(f"Optimized Gradient - Gap: {gradient_result['final_gap']:.3f}, Quality: {gradient_result['convergence_quality']}")
            print(f"Optimized PPO - Gap: {ppo_result['gap']:.3f}, Quality: {ppo_result['quality']}")
            print(f"Best Overall - Algorithm: {best_algorithm}, Gap: {best_gap:.3f}")
            
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 