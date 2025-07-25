# Logging System Integration Guide

## Overview

This guide explains how to integrate the comprehensive logging system into existing tournament experiment scripts.

## Quick Start

### 1. Basic Integration

```python
from utils.logger import setup_experiment_logging

# Setup logging at the beginning of your script
logger = setup_experiment_logging(
    experiment_name="two_players_experiment",
    log_dir="logs",
    log_level="INFO",
    console_output=True
)

# Use throughout your script
logger.info("Starting experiment...")
```

### 2. Experiment Lifecycle Logging

```python
# Log experiment start
config = {
    "algorithm": "PPO",
    "episodes": 10000,
    "learning_rate": 0.0003,
    "q_value": 25.0,
    "effort_range": [0, 100]
}
logger.log_experiment_start("two_players", config)

# ... run experiment ...

# Log experiment end
results = {
    "final_effort": 87.2,
    "theoretical_effort": 87.5,
    "gap": 0.3,
    "convergence_quality": "Excellent"
}
logger.log_experiment_end("two_players", results, success=True)
```

## Integration Examples

### PPO Agent Integration

```python
# In PPO training loop
for episode in range(num_episodes):
    # ... training code ...
    
    # Log training progress
    if episode % 1000 == 0:
        metrics = {
            "effort": current_effort,
            "reward": current_reward,
            "loss": current_loss
        }
        logger.log_training_progress("PPO", episode, metrics)
    
    # Log convergence checks
    if episode % convergence_check_interval == 0:
        logger.log_convergence_check(
            algorithm="PPO",
            episode=episode,
            current_effort=current_effort,
            theoretical_effort=theoretical_effort,
            gap=gap,
            std=std,
            quality=quality
        )
```

### Error Handling Integration

```python
try:
    # Risky operation
    result = some_calculation()
except Exception as e:
    context = {
        "function": "some_calculation",
        "algorithm": "PPO",
        "episode": episode,
        "parameters": {"effort": effort, "q_value": q_value}
    }
    logger.log_error_with_context(e, context)
    raise  # Re-raise if needed
```

### Performance Metrics Integration

```python
# After algorithm completion
metrics = {
    "convergence_time": end_time - start_time,
    "final_gap": abs(final_effort - theoretical_effort),
    "convergence_quality": quality,
    "episodes": total_episodes,
    "final_effort": final_effort
}
logger.log_algorithm_performance("PPO", metrics)
```

## Migration Guide for Existing Scripts

### Step 1: Add Logging Import

```python
# Add to imports
from utils.logger import setup_experiment_logging
```

### Step 2: Initialize Logger

```python
# Add at the beginning of main() function
def main():
    # Setup logging
    logger = setup_experiment_logging(
        experiment_name="your_experiment_name",
        log_dir="logs",
        log_level="INFO"
    )
    
    # Existing code...
```

### Step 3: Replace Print Statements

```python
# Replace:
print("Starting PPO experiment...")

# With:
logger.info("🧪 Starting PPO experiment...")
```

### Step 4: Add Experiment Tracking

```python
# Before experiment
logger.log_experiment_start(experiment_type, config)

# After experiment
logger.log_experiment_end(experiment_type, results, success=True)
```

### Step 5: Add Performance Logging

```python
# After each algorithm
logger.log_algorithm_performance(algorithm_name, metrics)
```

### Step 6: Close Logger

```python
# At the end of main()
logger.close()
```

## Log File Structure

The logging system creates several types of log files:

```
logs/
├── experiment_name_main.log                    # Main application logs
├── experiment_name_experiments_YYYYMMDD_HHMMSS.log  # Structured experiment data (JSON)
├── experiment_name_performance_YYYYMMDD_HHMMSS.log  # Performance metrics (JSON)
└── experiment_name_debug_YYYYMMDD_HHMMSS.log        # Debug information
```

## Log Analysis

### Using Command Line Tools

```bash
# Monitor real-time logs
tail -f logs/experiment_name_main.log

# Parse JSON logs with jq
cat logs/experiment_name_experiments_*.log | jq '.event'

# Filter specific algorithms
grep "PPO" logs/experiment_name_main.log

# Count convergence events
grep "convergence_check" logs/experiment_name_debug_*.log | wc -l
```

### Using Python for Analysis

```python
import json
import glob

# Load experiment logs
log_files = glob.glob("logs/*_experiments_*.log")
experiments = []

for file in log_files:
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                timestamp, data = line.split(' | ', 1)
                experiments.append(json.loads(data))

# Analyze results
for exp in experiments:
    if exp['event'] == 'experiment_end':
        print(f"Experiment: {exp['experiment_type']}")
        print(f"Success: {exp['success']}")
        print(f"Results: {exp['results']}")
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about execution
- **WARNING**: Unexpected but not critical events
- **ERROR**: Error conditions
- **CRITICAL**: Serious errors that might stop execution

### 2. Structure Your Logs

```python
# Good: Structured and informative
logger.info(f"🎯 {algorithm} converged: effort={effort:.2f}, gap={gap:.3f}")

# Avoid: Unstructured
logger.info("done")
```

### 3. Use Context in Error Logging

```python
# Good: Rich context
context = {
    "function": "calculate_utility",
    "algorithm": "PPO",
    "episode": 1500,
    "parameters": {"effort": effort, "q_value": q_value}
}
logger.log_error_with_context(error, context)

# Avoid: Minimal context
logger.error(str(error))
```

### 4. Log Configuration Changes

```python
# When updating algorithm parameters
logger.log_configuration_change("PPOAgent", old_config, new_config)
```

### 5. Close Loggers Properly

```python
# Always close at the end
try:
    # Main experiment code
    pass
finally:
    logger.close()
```

## Performance Considerations

1. **Log Level**: Use appropriate log levels to control verbosity
2. **File Rotation**: Logs automatically rotate when they reach size limits
3. **JSON Logs**: Structured logs are slightly more expensive but much more useful
4. **Console Output**: Can be disabled for better performance in production

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure the logs directory is writable
2. **Large Log Files**: Adjust `max_file_size` and `backup_count` parameters
3. **Too Verbose**: Increase log level (DEBUG → INFO → WARNING → ERROR)
4. **Missing Logs**: Check that logger.close() is called

### Configuration Options

```python
logger = setup_experiment_logging(
    experiment_name="my_experiment",
    log_dir="custom_logs",           # Custom log directory
    log_level="WARNING",             # Higher threshold
    max_file_size=50 * 1024 * 1024, # 50MB files
    backup_count=10,                 # Keep 10 backups
    console_output=False             # Disable console output
)
```

## Examples in Existing Scripts

See the following files for integration examples:
- `utils/logging_example.py` - Complete demonstration
- `run/run_two_players.py` - (To be updated with logging)
- `run/run_three_players.py` - (To be updated with logging)
- `agents/ppo_agent.py` - (To be updated with logging) 