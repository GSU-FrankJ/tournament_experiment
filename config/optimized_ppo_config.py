# Optimized PPO Configuration
# Based on successful random search results (Trial 15)
# Achieves gap of 0.066 from theoretical optimum

OPTIMIZED_PPO_CONFIG = {
    # Core learning parameters
    'learning_rate': 0.0001,
    'gamma': 0.995,  # Higher discount factor - key for long-term rewards
    'gae_lambda': 0.97,
    
    # Network architecture
    'hidden_dim': 128,
    'num_layers': 3,
    'activation': 'tanh',  # Critical: tanh outperforms relu
    'use_layer_norm': False,
    'use_residual': False,
    'separate_networks': True,
    'dropout_rate': 0.05,
    
    # PPO-specific parameters
    'clip_epsilon': 0.2,
    'value_coef': 0.75,  # Higher value network weight
    'entropy_coef': 0.005,  # Moderate entropy for exploration
    'max_grad_norm': 0.3,  # Stricter gradient clipping
    
    # Training parameters
    'batch_size': 64,  # Smaller batch for better gradients
    'update_epochs': 10,
    'weight_decay': 1e-05,
    
    # Learning rate scheduling
    'lr_schedule': 'cosine_annealing',
    'lr_decay_factor': 0.95,
    'lr_patience': 3000,
    
    # Normalization
    'reward_normalization': True,
    'observation_normalization': False,
    
    # Training settings
    'max_episodes': 50000,
    'convergence_threshold': 2.0,  # Gap from theoretical value
    'patience': 5000,
    'log_interval': 1000,
    'early_success_threshold': 2.0
}

# Performance expectations based on optimization results
EXPECTED_PERFORMANCE = {
    'target_effort': 87.5,  # Theoretical optimum
    'achieved_effort': 87.57,  # Best result
    'gap': 0.066,
    'convergence_episodes': 10000,
    'quality': 'Fair'
}

# Key insights from optimization
OPTIMIZATION_INSIGHTS = {
    'critical_parameters': [
        'activation=tanh',
        'gamma=0.995', 
        'value_coef=0.75',
        'batch_size=64',
        'lr_schedule=cosine_annealing'
    ],
    'performance_factors': {
        'tanh_vs_relu': 'tanh provides better gradient flow for this problem',
        'high_gamma': 'Long-term reward consideration crucial for equilibrium',
        'small_batch': 'Better gradient estimates with smaller batches',
        'cosine_annealing': 'Helps fine-tuning near convergence',
        'value_weight': 'Higher value coefficient improves stability'
    }
} 