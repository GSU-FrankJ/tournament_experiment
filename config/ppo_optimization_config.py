# PPO Optimization Configuration
# This file defines hyperparameter search spaces and optimization settings

import numpy as np

# Hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACE = {
    # Learning rate search space
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    
    # Network architecture parameters
    'hidden_dim': [64, 128, 256, 512],
    'num_layers': [2, 3, 4],
    'activation': ['relu', 'tanh', 'elu'],
    
    # PPO-specific parameters
    'clip_epsilon': [0.1, 0.15, 0.2, 0.25, 0.3],
    'value_coef': [0.25, 0.5, 0.75, 1.0],
    'entropy_coef': [0.001, 0.005, 0.01, 0.02, 0.05],
    'max_grad_norm': [0.3, 0.5, 1.0, 2.0],
    
    # Training parameters
    'batch_size': [32, 64, 128, 256],
    'update_epochs': [4, 6, 8, 10, 12],
    'gamma': [0.95, 0.97, 0.99, 0.995],
    'gae_lambda': [0.9, 0.95, 0.97, 0.99],
    
    # Regularization
    'weight_decay': [0, 1e-6, 1e-5, 1e-4],
    'dropout_rate': [0.0, 0.05, 0.1, 0.15],
    
    # Learning rate scheduling
    'lr_schedule': ['constant', 'linear_decay', 'cosine_annealing', 'reduce_on_plateau'],
    'lr_decay_factor': [0.8, 0.9, 0.95],
    'lr_patience': [1000, 2000, 3000],
    
    # Advanced techniques
    'use_layer_norm': [True, False],
    'use_residual': [True, False],
    'separate_networks': [True, False],
    'reward_normalization': [True, False],
    'observation_normalization': [True, False],
}

# Optimization settings
OPTIMIZATION_CONFIG = {
    'max_trials': 100,  # Maximum number of hyperparameter combinations to try
    'max_episodes_per_trial': 15000,  # Maximum episodes per trial
    'early_stopping_patience': 3000,  # Episodes without improvement before stopping
    'convergence_threshold': 2.0,  # Gap from theoretical value for early success
    'min_episodes_before_eval': 2000,  # Minimum episodes before evaluating performance
    'eval_interval': 1000,  # Episodes between performance evaluations
    'target_performance': 82.5,  # Target performance (gap < 5.0 from 87.5)
    'num_seeds': 3,  # Number of random seeds to test best configurations
    'parallel_trials': 4,  # Number of parallel trials (if using distributed optimization)
}

# Bayesian optimization settings (for advanced search)
BAYESIAN_CONFIG = {
    'n_initial_points': 20,  # Random points before Bayesian optimization
    'acquisition_function': 'EI',  # Expected Improvement
    'kappa': 2.576,  # Exploration parameter for UCB
    'xi': 0.01,  # Exploration parameter for EI
}

# Performance tracking
TRACKING_CONFIG = {
    'log_dir': 'results/optimization_logs',
    'checkpoint_dir': 'results/optimization_checkpoints',
    'tensorboard_dir': 'results/tensorboard',
    'save_best_models': True,
    'save_all_configs': True,
    'detailed_logging': True,
}

# Default baseline configuration (current best known)
BASELINE_CONFIG = {
    'learning_rate': 1e-4,
    'hidden_dim': 128,
    'num_layers': 3,
    'activation': 'relu',
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'batch_size': 128,
    'update_epochs': 8,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'weight_decay': 1e-5,
    'dropout_rate': 0.1,
    'lr_schedule': 'reduce_on_plateau',
    'lr_decay_factor': 0.9,
    'lr_patience': 1500,
    'use_layer_norm': True,
    'use_residual': False,
    'separate_networks': True,
    'reward_normalization': True,
    'observation_normalization': False,
}

# Quick search space for initial testing (smaller subset)
QUICK_SEARCH_SPACE = {
    'learning_rate': [5e-5, 1e-4, 5e-4],
    'clip_epsilon': [0.15, 0.2, 0.25],
    'entropy_coef': [0.005, 0.01, 0.02],
    'update_epochs': [6, 8, 10],
    'gamma': [0.97, 0.99],
    'hidden_dim': [128, 256],
}

# Function to generate random configuration
def generate_random_config():
    """Generate a random configuration from the search space"""
    config = {}
    for param, values in HYPERPARAMETER_SEARCH_SPACE.items():
        if isinstance(values, list):
            config[param] = np.random.choice(values)
        else:
            config[param] = values
    return config

# Function to generate grid search configurations
def generate_grid_configs(search_space=None, max_configs=50):
    """Generate configurations for grid search"""
    if search_space is None:
        search_space = QUICK_SEARCH_SPACE
    
    import itertools
    
    # Get all parameter combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    configs = []
    for combination in itertools.product(*param_values):
        if len(configs) >= max_configs:
            break
        config = dict(zip(param_names, combination))
        # Fill in missing parameters with baseline values
        for param, value in BASELINE_CONFIG.items():
            if param not in config:
                config[param] = value
        configs.append(config)
    
    return configs

# Function to validate configuration
def validate_config(config):
    """Validate that a configuration has all required parameters"""
    required_params = set(BASELINE_CONFIG.keys())
    config_params = set(config.keys())
    
    missing_params = required_params - config_params
    if missing_params:
        raise ValueError(f"Configuration missing required parameters: {missing_params}")
    
    return True 