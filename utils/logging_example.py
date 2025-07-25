#!/usr/bin/env python3
"""
Logging System Example

This script demonstrates how to use the comprehensive logging system
for tournament experiments.
"""

import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import ExperimentLogger, setup_experiment_logging, log_info, log_debug, log_warning, log_error

def demonstrate_basic_logging():
    """Demonstrate basic logging functionality"""
    print("=== Basic Logging Demonstration ===")
    
    # Setup logger
    logger = setup_experiment_logging(
        experiment_name="demo_experiment",
        log_dir="logs/demo",
        log_level="DEBUG",
        console_output=True
    )
    
    # Basic logging levels
    logger.debug("🔍 This is a debug message - detailed information for developers")
    logger.info("ℹ️  This is an info message - general information about execution")
    logger.warning("⚠️  This is a warning message - something unexpected but not critical")
    logger.error("❌ This is an error message - something went wrong")
    logger.critical("🚨 This is a critical message - system-level failure")
    
    # Convenience functions
    log_info("📢 Using convenience function for quick logging")
    log_debug("🔧 Debug info via convenience function")
    log_warning("⚠️  Warning via convenience function")
    log_error("💥 Error via convenience function")
    
    return logger

def demonstrate_experiment_logging(logger: ExperimentLogger):
    """Demonstrate experiment-specific logging"""
    print("\n=== Experiment Logging Demonstration ===")
    
    # Log experiment start
    config = {
        "algorithm": "PPO",
        "episodes": 1000,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "environment": "two_players",
        "q_value": 25.0,
        "effort_range": [0, 100]
    }
    
    logger.log_experiment_start("two_players_demo", config)
    
    # Simulate experiment progress
    for episode in range(0, 1001, 200):
        metrics = {
            "effort": 85.0 + random.uniform(-5, 5),
            "reward": random.uniform(0.5, 1.0),
            "loss": random.uniform(0.1, 0.5),
            "std": random.uniform(1.0, 3.0)
        }
        logger.log_training_progress("PPO", episode, metrics)
        time.sleep(0.1)  # Simulate processing time
    
    # Log convergence check
    logger.log_convergence_check(
        algorithm="PPO",
        episode=1000,
        current_effort=87.2,
        theoretical_effort=87.5,
        gap=0.3,
        std=1.8,
        quality="Excellent"
    )
    
    # Log experiment end
    results = {
        "final_effort": 87.2,
        "theoretical_effort": 87.5,
        "gap": 0.3,
        "convergence_quality": "Excellent",
        "total_episodes": 1000,
        "training_time": 45.2
    }
    
    logger.log_experiment_end("two_players_demo", results, success=True)

def demonstrate_performance_logging(logger: ExperimentLogger):
    """Demonstrate performance metrics logging"""
    print("\n=== Performance Logging Demonstration ===")
    
    # Log algorithm performance for different algorithms
    algorithms = ["Gradient", "PPO", "Enhanced_PPO"]
    
    for algorithm in algorithms:
        # Simulate different performance metrics
        if algorithm == "Gradient":
            metrics = {
                "convergence_time": 0.45,
                "final_gap": 0.001,
                "convergence_quality": "Excellent",
                "iterations": 8070,
                "final_effort": 87.499,
                "stability": "Perfect"
            }
        elif algorithm == "PPO":
            metrics = {
                "convergence_time": 25.8,
                "final_gap": 2.3,
                "convergence_quality": "Good",
                "episodes": 5000,
                "final_effort": 85.2,
                "stability": "Stable"
            }
        else:  # Enhanced_PPO
            metrics = {
                "convergence_time": 18.5,
                "final_gap": 0.8,
                "convergence_quality": "Excellent",
                "episodes": 3500,
                "final_effort": 86.7,
                "stability": "Very Stable"
            }
        
        logger.log_algorithm_performance(algorithm, metrics)
        time.sleep(0.1)

def demonstrate_error_logging(logger: ExperimentLogger):
    """Demonstrate error logging with context"""
    print("\n=== Error Logging Demonstration ===")
    
    # Simulate various types of errors
    try:
        # Simulate a division by zero error
        result = 1 / 0
    except ZeroDivisionError as e:
        context = {
            "function": "calculate_utility",
            "algorithm": "PPO",
            "episode": 1500,
            "parameters": {"effort": 0, "q_value": 25.0}
        }
        logger.log_error_with_context(e, context)
    
    try:
        # Simulate a key error
        config = {"learning_rate": 0.001}
        hidden_dim = config["hidden_dim"]  # This key doesn't exist
    except KeyError as e:
        context = {
            "function": "initialize_agent",
            "algorithm": "PPO",
            "config": config
        }
        logger.log_error_with_context(e, context)
    
    try:
        # Simulate a value error
        effort = float("invalid_number")
    except ValueError as e:
        context = {
            "function": "parse_effort_value",
            "input": "invalid_number",
            "expected_type": "float"
        }
        logger.log_error_with_context(e, context)

def demonstrate_configuration_logging(logger: ExperimentLogger):
    """Demonstrate configuration change logging"""
    print("\n=== Configuration Change Logging Demonstration ===")
    
    # Simulate configuration changes
    old_ppo_config = {
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "num_layers": 2,
        "activation": "relu"
    }
    
    new_ppo_config = {
        "learning_rate": 0.0003,
        "hidden_dim": 128,
        "num_layers": 3,
        "activation": "tanh"
    }
    
    logger.log_configuration_change("PPOAgent", old_ppo_config, new_ppo_config)
    
    # Environment configuration change
    old_env_config = {
        "q_value": 25.0,
        "effort_range": [0, 100],
        "num_players": 2
    }
    
    new_env_config = {
        "q_value": 40.0,
        "effort_range": [0, 200],
        "num_players": 2
    }
    
    logger.log_configuration_change("Environment", old_env_config, new_env_config)

def demonstrate_log_analysis():
    """Demonstrate how to analyze log files"""
    print("\n=== Log Analysis Demonstration ===")
    
    log_dir = "logs/demo"
    
    if os.path.exists(log_dir):
        print(f"📁 Log files created in: {os.path.abspath(log_dir)}")
        
        # List all log files
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  📄 {file} ({file_size} bytes)")
        
        print("\n💡 Log file types:")
        print("  • *_main.log: Main application logs with all levels")
        print("  • *_experiments_*.log: Structured experiment data (JSON)")
        print("  • *_performance_*.log: Algorithm performance metrics (JSON)")
        print("  • *_debug_*.log: Detailed debug information")
        
        print("\n🔍 To analyze logs:")
        print("  • Use 'tail -f' to monitor real-time logs")
        print("  • Use 'jq' to parse JSON-formatted experiment/performance logs")
        print("  • Use 'grep' to filter specific events or algorithms")
        print("  • Import JSON logs into analysis tools like Jupyter notebooks")
    else:
        print("❌ No log directory found. Run the logging demonstration first.")

def main():
    """Main demonstration function"""
    print("🚀 Tournament Experiment Logging System Demonstration")
    print("=" * 60)
    
    try:
        # Basic logging demonstration
        logger = demonstrate_basic_logging()
        
        # Experiment-specific logging
        demonstrate_experiment_logging(logger)
        
        # Performance logging
        demonstrate_performance_logging(logger)
        
        # Error logging
        demonstrate_error_logging(logger)
        
        # Configuration logging
        demonstrate_configuration_logging(logger)
        
        # Close logger
        logger.close()
        
        # Log analysis
        demonstrate_log_analysis()
        
        print("\n✅ Logging demonstration completed successfully!")
        print("📁 Check the 'logs/demo' directory for generated log files")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 