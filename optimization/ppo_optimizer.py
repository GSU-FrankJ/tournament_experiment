import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from pathlib import Path

from config.ppo_optimization_config import (
    HYPERPARAMETER_SEARCH_SPACE, OPTIMIZATION_CONFIG, BAYESIAN_CONFIG, 
    TRACKING_CONFIG, BASELINE_CONFIG, QUICK_SEARCH_SPACE,
    generate_random_config, generate_grid_configs, validate_config
)
from config.one_stage_two_players import config as game_config
from envs.one_stage_env import OneStageEnv
from agents.optimized_ppo_agent import OptimizedPPOAgent

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class PPOOptimizer:
    """
    Comprehensive PPO hyperparameter optimization framework
    Supports grid search, random search, and Bayesian optimization
    """
    
    def __init__(self, optimization_type='grid', use_quick_search=True):
        self.optimization_type = optimization_type
        self.use_quick_search = use_quick_search
        self.theoretical_effort = game_config["effort"]
        
        # Create directories
        self.setup_directories()
        
        # Initialize tracking
        self.results = []
        self.best_config = None
        self.best_performance = float('inf')  # Gap from theoretical value
        
        # Load existing results if available
        self.load_existing_results()
        
    def setup_directories(self):
        """Create necessary directories for logging and checkpoints"""
        for dir_path in TRACKING_CONFIG.values():
            if isinstance(dir_path, str) and 'dir' in dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_existing_results(self):
        """Load existing optimization results to avoid re-running"""
        results_file = Path(TRACKING_CONFIG['log_dir']) / 'optimization_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    self.results = data.get('results', [])
                    self.best_config = data.get('best_config', None)
                    self.best_performance = data.get('best_performance', float('inf'))
                print(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                print(f"Error loading existing results: {e}")
    
    def save_results(self):
        """Save optimization results to file"""
        results_file = Path(TRACKING_CONFIG['log_dir']) / 'optimization_results.json'
        
        # Convert numpy types to Python types
        converted_results = convert_numpy_types(self.results)
        converted_best_config = convert_numpy_types(self.best_config)
        converted_best_performance = convert_numpy_types(self.best_performance)
        
        data = {
            'results': converted_results,
            'best_config': converted_best_config,
            'best_performance': converted_best_performance,
            'timestamp': datetime.now().isoformat(),
            'optimization_type': self.optimization_type
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save as CSV for easy analysis
        if self.results:
            # Flatten results for CSV
            flattened_results = []
            for result in self.results:
                flat_result = {
                    'trial_id': result.get('trial_id'),
                    'final_effort': result.get('final_effort'),
                    'final_gap': result.get('final_gap'),
                    'convergence_quality': result.get('convergence_quality'),
                    'episodes_trained': result.get('episodes_trained'),
                    'training_time': result.get('training_time'),
                    'success': result.get('success', False)
                }
                # Add config parameters
                config = result.get('config', {})
                for param, value in config.items():
                    flat_result[f'config_{param}'] = convert_numpy_types(value)
                
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            csv_file = Path(TRACKING_CONFIG['log_dir']) / 'optimization_results.csv'
            df.to_csv(csv_file, index=False)
    
    def evaluate_configuration(self, config: Dict, trial_id: int) -> Dict:
        """
        Evaluate a single hyperparameter configuration
        Returns performance metrics and training statistics
        """
        print(f"\n=== Trial {trial_id}: Evaluating Configuration ===")
        print(f"Config: {config}")
        
        start_time = time.time()
        
        try:
            # Create environment
            env = OneStageEnv(game_config)
            
            # Create optimized PPO agent with this configuration
            agent = OptimizedPPOAgent(
                config=config,
                effort_range=game_config["effort_range"],
                theoretical_effort=self.theoretical_effort,
                log_path=f"{TRACKING_CONFIG['log_dir']}/trial_{trial_id}_agent.csv"
            )
            
            # Training loop
            num_episodes = OPTIMIZATION_CONFIG['max_episodes_per_trial']
            patience = OPTIMIZATION_CONFIG['early_stopping_patience']
            eval_interval = OPTIMIZATION_CONFIG['eval_interval']
            min_episodes = OPTIMIZATION_CONFIG['min_episodes_before_eval']
            
            best_gap = float('inf')
            episodes_without_improvement = 0
            training_metrics = []
            
            for episode in range(num_episodes):
                # Run episode
                state1, state2 = env.reset()
                a1 = agent.select_action(state1)
                a2 = agent.select_action(state2)  # For symmetry, though we focus on agent1
                
                _, rewards, _, _, info = env.step(torch.stack([a1, a2]))
                agent.store_reward(rewards[0])
                agent.update_policy(episode=episode, last_effort=a1)
                
                # Evaluate performance periodically
                if episode % eval_interval == 0 and episode >= min_episodes:
                    stats = agent.get_convergence_stats()
                    if stats:
                        current_effort = stats['recent_mean_effort']
                        effort_std = stats['recent_std_effort']
                        gap = abs(current_effort - self.theoretical_effort)
                        
                        training_metrics.append({
                            'episode': episode,
                            'effort': current_effort,
                            'std': effort_std,
                            'gap': gap
                        })
                        
                        print(f"  Episode {episode}: effort={current_effort:.2f}±{effort_std:.2f}, gap={gap:.3f}")
                        
                        # Check for improvement
                        if gap < best_gap:
                            best_gap = gap
                            episodes_without_improvement = 0
                        else:
                            episodes_without_improvement += eval_interval
                        
                        # Early stopping conditions
                        if gap < OPTIMIZATION_CONFIG['convergence_threshold']:
                            print(f"  Early success! Gap {gap:.3f} < threshold {OPTIMIZATION_CONFIG['convergence_threshold']}")
                            break
                        
                        if episodes_without_improvement >= patience:
                            print(f"  Early stopping: no improvement for {patience} episodes")
                            break
            
            # Final evaluation
            final_stats = agent.get_convergence_stats()
            if final_stats:
                final_effort = final_stats['recent_mean_effort']
                final_std = final_stats['recent_std_effort']
                final_gap = abs(final_effort - self.theoretical_effort)
            else:
                # Fallback to last episode
                final_effort = info["efforts"][0].item()
                final_std = 0.0
                final_gap = abs(final_effort - self.theoretical_effort)
            
            training_time = time.time() - start_time
            
            # Determine convergence quality
            if final_gap < 2.0 and final_std < 3.0:
                quality = "Excellent"
            elif final_gap < 5.0 and final_std < 5.0:
                quality = "Good"
            elif final_gap < 10.0:
                quality = "Fair"
            else:
                quality = "Poor"
            
            result = {
                'trial_id': trial_id,
                'config': config,
                'final_effort': final_effort,
                'final_std': final_std,
                'final_gap': final_gap,
                'best_gap': best_gap,
                'episodes_trained': episode + 1,
                'training_time': training_time,
                'convergence_quality': quality,
                'training_metrics': training_metrics,
                'success': final_gap < OPTIMIZATION_CONFIG['convergence_threshold']
            }
            
            print(f"  Final result: effort={final_effort:.2f}, gap={final_gap:.3f}, quality={quality}")
            
            return result
            
        except Exception as e:
            print(f"  Error in trial {trial_id}: {e}")
            return {
                'trial_id': trial_id,
                'config': config,
                'error': str(e),
                'final_gap': float('inf'),
                'success': False
            }
    
    def run_grid_search(self, max_trials: Optional[int] = None) -> List[Dict]:
        """Run grid search optimization"""
        print("=== Starting Grid Search Optimization ===")
        
        # Generate configurations
        search_space = QUICK_SEARCH_SPACE if self.use_quick_search else HYPERPARAMETER_SEARCH_SPACE
        configs = generate_grid_configs(search_space, max_configs=max_trials or OPTIMIZATION_CONFIG['max_trials'])
        
        print(f"Generated {len(configs)} configurations to test")
        
        # Run trials
        for i, config in enumerate(configs):
            trial_id = len(self.results) + 1
            
            # Skip if already tested (based on config hash)
            config_hash = hash(str(sorted(config.items())))
            if any(r.get('config_hash') == config_hash for r in self.results):
                print(f"Skipping trial {trial_id} (already tested)")
                continue
            
            result = self.evaluate_configuration(config, trial_id)
            result['config_hash'] = config_hash
            result['optimization_type'] = 'grid_search'
            
            self.results.append(result)
            
            # Update best configuration
            if result.get('final_gap', float('inf')) < self.best_performance:
                self.best_performance = result['final_gap']
                self.best_config = config.copy()
                print(f"*** New best configuration! Gap: {self.best_performance:.3f} ***")
            
            # Save progress
            self.save_results()
            
            # Early termination if target reached
            if result.get('final_gap', float('inf')) < (87.5 - OPTIMIZATION_CONFIG['target_performance']):
                print(f"Target performance reached! Stopping optimization.")
                break
        
        return self.results
    
    def run_random_search(self, num_trials: Optional[int] = None) -> List[Dict]:
        """Run random search optimization"""
        print("=== Starting Random Search Optimization ===")
        
        num_trials = num_trials or OPTIMIZATION_CONFIG['max_trials']
        
        for i in range(num_trials):
            trial_id = len(self.results) + 1
            
            # Generate random configuration
            config = generate_random_config()
            
            # Fill in missing parameters with baseline
            for param, value in BASELINE_CONFIG.items():
                if param not in config:
                    config[param] = value
            
            result = self.evaluate_configuration(config, trial_id)
            result['optimization_type'] = 'random_search'
            
            self.results.append(result)
            
            # Update best configuration
            if result.get('final_gap', float('inf')) < self.best_performance:
                self.best_performance = result['final_gap']
                self.best_config = config.copy()
                print(f"*** New best configuration! Gap: {self.best_performance:.3f} ***")
            
            # Save progress
            self.save_results()
            
            # Early termination if target reached
            if result.get('final_gap', float('inf')) < (87.5 - OPTIMIZATION_CONFIG['target_performance']):
                print(f"Target performance reached! Stopping optimization.")
                break
        
        return self.results
    
    def analyze_results(self) -> Dict:
        """Analyze optimization results and provide insights"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = [r for r in self.results if not r.get('error') and r.get('final_gap', float('inf')) < float('inf')]
        
        if not successful_results:
            return {"error": "No successful trials"}
        
        # Sort by performance
        successful_results.sort(key=lambda x: x.get('final_gap', float('inf')))
        
        # Top configurations
        top_configs = successful_results[:5]
        
        # Performance statistics
        gaps = [r['final_gap'] for r in successful_results]
        
        analysis = {
            'total_trials': len(self.results),
            'successful_trials': len(successful_results),
            'best_gap': min(gaps),
            'mean_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'top_5_configs': top_configs,
            'best_config': self.best_config,
            'target_achieved': min(gaps) < (87.5 - OPTIMIZATION_CONFIG['target_performance'])
        }
        
        return analysis
    
    def run_optimization(self) -> Dict:
        """Run the specified optimization type"""
        if self.optimization_type == 'grid':
            self.run_grid_search()
        elif self.optimization_type == 'random':
            self.run_random_search()
        else:
            raise ValueError(f"Unknown optimization type: {self.optimization_type}")
        
        return self.analyze_results()

def main():
    """Main function to run PPO optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Hyperparameter Optimization')
    parser.add_argument('--type', choices=['grid', 'random'], default='grid',
                       help='Optimization type')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick search space (fewer parameters)')
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Maximum number of trials')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = PPOOptimizer(
        optimization_type=args.type,
        use_quick_search=args.quick
    )
    
    # Run optimization
    print(f"Starting {args.type} optimization...")
    results = optimizer.run_optimization()
    
    # Print summary
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total trials: {results['total_trials']}")
    print(f"Successful trials: {results['successful_trials']}")
    print(f"Best gap: {results['best_gap']:.3f}")
    print(f"Target achieved: {results['target_achieved']}")
    
    if results['best_config']:
        print(f"\nBest configuration:")
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")

if __name__ == "__main__":
    main() 