import csv
import os
import logging
import logging.handlers
import json
import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys

def save_result(row_dict, filename):
    """
    Save experimental results to CSV file (legacy function for backward compatibility)
    
    Args:
        row_dict: Dictionary containing result data
        filename: Path to CSV file
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def save_standardized_result(experiment_data: Dict[str, Any], filename: str):
    """
    Save experimental results using standardized table header format
    
    Standard header format:
    stage1_weight,stage2_weight,k1,k2,information_revelation,theoretical_stage1_effort,theoretical_stage2_effort,Model_training,final_stage1_effort,final_stage2_effort,final_weighted_effort,Convergence_Quality,episodes,Gap_from_theoretical
    
    Args:
        experiment_data: Dictionary containing experiment-specific data
        filename: Path to CSV file
    """
    # Map experiment data to standard format
    standardized_row = {
        # Run identification columns (for sweep correlation)
        "run_id": experiment_data.get("run_id", ""),
        "variant_name": experiment_data.get("variant_name", "baseline"),
        "stage1_weight": experiment_data.get("stage1_weight", 1.0),
        "stage2_weight": experiment_data.get("stage2_weight", 0.0),
        "k1": experiment_data.get("k1", experiment_data.get("k", 0.0004)),
        "k2": experiment_data.get("k2", experiment_data.get("k", 0.0004)),
        "information_revelation": experiment_data.get("information_revelation", "none"),
        "theoretical_stage1_effort": experiment_data.get("theoretical_stage1_effort", 
                                                        experiment_data.get("theoretical_effort", 0.0)),
        "theoretical_stage2_effort": experiment_data.get("theoretical_stage2_effort", 0.0),
        "Model_training": experiment_data.get("Model_training", experiment_data.get("algorithm", "Unknown")),
        "final_stage1_effort": experiment_data.get("final_stage1_effort", 
                                                  experiment_data.get("final_effort", 
                                                                     experiment_data.get("actual_effort", 0.0))),
        "final_stage2_effort": experiment_data.get("final_stage2_effort", 0.0),
        "final_weighted_effort": experiment_data.get("final_weighted_effort",
                                                    experiment_data.get("final_stage1_effort", 
                                                                       experiment_data.get("final_effort",
                                                                                          experiment_data.get("actual_effort", 0.0)))),
        "Convergence_Quality": experiment_data.get("Convergence_Quality", 
                                                  experiment_data.get("quality", "Unknown")),
        "episodes": experiment_data.get("episodes", 
                                       experiment_data.get("convergence_time", "N/A")),
        "Gap_from_theoretical": experiment_data.get("Gap_from_theoretical",
                                                   experiment_data.get("gap", 0.0)),
        "abs_err": experiment_data.get("abs_err", ""),
        "stage2_gap_unweighted": experiment_data.get("stage2_gap_unweighted", ""),
        "gradient_iterations": experiment_data.get("gradient_iterations", ""),
        "gradient_final_grad": experiment_data.get("gradient_final_grad", ""),
        "gradient_mode": experiment_data.get("gradient_mode", ""),
        "opp_mode": experiment_data.get("opp_mode", ""),
        "opp_sync_interval": experiment_data.get("opp_sync_interval", ""),
        "opp_ema_tau": experiment_data.get("opp_ema_tau", ""),
        "opp_hist_size": experiment_data.get("opp_hist_size", ""),
        "last_sync_step": experiment_data.get("last_sync_step", ""),
        "approx_kl": experiment_data.get("approx_kl", ""),
        "batch_entropy": experiment_data.get("batch_entropy", ""),
        "alpha_mean": experiment_data.get("alpha_mean", ""),
        "beta_mean": experiment_data.get("beta_mean", ""),
        "eval_vs_opponent_effort": experiment_data.get("eval_vs_opponent_effort", ""),
        "eval_vs_opponent_reward": experiment_data.get("eval_vs_opponent_reward", ""),
        "eval_vs_opponent_opp_effort": experiment_data.get("eval_vs_opponent_opp_effort", ""),
        "eval_vs_opponent_abs_err": experiment_data.get("eval_vs_opponent_abs_err", ""),
        "eval_vs_history_effort_mean": experiment_data.get("eval_vs_history_effort_mean", ""),
        "eval_vs_history_effort_std": experiment_data.get("eval_vs_history_effort_std", ""),
        "eval_vs_history_reward_mean": experiment_data.get("eval_vs_history_reward_mean", ""),
        "eval_vs_history_reward_std": experiment_data.get("eval_vs_history_reward_std", ""),
        "eval_vs_history_abs_err_mean": experiment_data.get("eval_vs_history_abs_err_mean", ""),
    }
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV with standard header
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        fieldnames = [
            "run_id", "variant_name",
            "stage1_weight", "stage2_weight", "k1", "k2", "information_revelation",
            "theoretical_stage1_effort", "theoretical_stage2_effort", "Model_training",
            "final_stage1_effort", "final_stage2_effort", "final_weighted_effort",
            "Convergence_Quality", "episodes", "Gap_from_theoretical", "abs_err", "stage2_gap_unweighted",
            "gradient_iterations", "gradient_final_grad", "gradient_mode",
            "opp_mode", "opp_sync_interval", "opp_ema_tau", "opp_hist_size", "last_sync_step",
            "approx_kl", "batch_entropy", "alpha_mean", "beta_mean",
            "kl_proxy_max", "kl_proxy_mean",
            "ratio_max", "ratio_mean",
            "clip_frac_max", "clip_frac_mean",
            "approx_kl_max_abs",
            "eval_vs_opponent_effort", "eval_vs_opponent_reward", "eval_vs_opponent_opp_effort",
            "eval_vs_opponent_abs_err", "eval_vs_history_effort_mean", "eval_vs_history_effort_std",
            "eval_vs_history_reward_mean", "eval_vs_history_reward_std", "eval_vs_history_abs_err_mean"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(standardized_row)

def create_experiment_result(
    algorithm: str,
    final_effort: float,
    theoretical_effort: float,
    convergence_quality: str,
    episodes: str = "N/A",
    stage1_weight: float = 1.0,
    stage2_weight: float = 0.0,
    k1: float = 0.0004,
    k2: float = 0.0004,
    information_revelation: str = "none",
    theoretical_stage2_effort: float = 0.0,
    final_stage2_effort: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized experiment result dictionary
    
    Args:
        algorithm: Algorithm name (e.g., "Gradient", "PPO", "Enhanced_PPO")
        final_effort: Final effort value achieved by algorithm
        theoretical_effort: Theoretical optimal effort value
        convergence_quality: Quality assessment ("Excellent", "Good", "Fair", "Poor")
        episodes: Training episodes or steps
        stage1_weight: Weight for stage 1 (default 1.0 for single-stage)
        stage2_weight: Weight for stage 2 (default 0.0 for single-stage)
        k1: Cost parameter for stage 1
        k2: Cost parameter for stage 2  
        information_revelation: Information structure ("none", "partial", "full")
        theoretical_stage2_effort: Theoretical effort for stage 2
        final_stage2_effort: Final effort for stage 2
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with standardized experiment result format
    """
    gap = abs(final_effort - theoretical_effort)
    final_weighted_effort = (stage1_weight * final_effort + 
                           stage2_weight * final_stage2_effort)
    
    return {
        "stage1_weight": stage1_weight,
        "stage2_weight": stage2_weight,
        "k1": k1,
        "k2": k2,
        "information_revelation": information_revelation,
        "theoretical_stage1_effort": theoretical_effort,
        "theoretical_stage2_effort": theoretical_stage2_effort,
        "Model_training": algorithm,
        "final_stage1_effort": round(final_effort, 2),
        "final_stage2_effort": round(final_stage2_effort, 2),
        "final_weighted_effort": round(final_weighted_effort, 2),
        "Convergence_Quality": convergence_quality,
        "episodes": episodes,
        "Gap_from_theoretical": round(gap, 3)
    }

class ExperimentLogger:
    """
    Comprehensive logging system for tournament experiments
    
    Features:
    - Hierarchical logging with multiple levels
    - Rotating file handlers for log management
    - JSON structured logging for experiment data
    - Console and file output
    - Experiment session tracking
    - Performance metrics logging
    """
    
    def __init__(self, 
                 experiment_name: str = "tournament_experiment",
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True):
        """
        Initialize the experiment logger
        
        Args:
            experiment_name: Name of the experiment for log file naming
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_file_size: Maximum size of each log file before rotation
            backup_count: Number of backup files to keep
            console_output: Whether to output logs to console
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session ID for this experiment run
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup loggers
        self._setup_main_logger(log_level, max_file_size, backup_count, console_output)
        self._setup_experiment_logger()
        self._setup_performance_logger()
        self._setup_debug_logger()
        
        # Log session start
        self.info(f"🚀 Experiment session started: {self.session_id}")
        self.info(f"📁 Log directory: {self.log_dir.absolute()}")
        
    def _setup_main_logger(self, log_level: str, max_file_size: int, backup_count: int, console_output: bool):
        """Setup the main application logger"""
        self.logger = logging.getLogger(f"{self.experiment_name}.main")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.experiment_name}_main.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _setup_experiment_logger(self):
        """Setup logger for experiment-specific data"""
        self.exp_logger = logging.getLogger(f"{self.experiment_name}.experiment")
        self.exp_logger.setLevel(logging.INFO)
        
        # JSON formatter for structured experiment data
        exp_formatter = logging.Formatter('%(asctime)s | %(message)s')
        
        exp_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.experiment_name}_experiments_{self.session_id}.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        exp_handler.setFormatter(exp_formatter)
        self.exp_logger.addHandler(exp_handler)
    
    def _setup_performance_logger(self):
        """Setup logger for performance metrics"""
        self.perf_logger = logging.getLogger(f"{self.experiment_name}.performance")
        self.perf_logger.setLevel(logging.INFO)
        
        perf_formatter = logging.Formatter('%(asctime)s | %(message)s')
        
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.experiment_name}_performance_{self.session_id}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        perf_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_handler)
    
    def _setup_debug_logger(self):
        """Setup logger for debug information"""
        self.debug_logger = logging.getLogger(f"{self.experiment_name}.debug")
        self.debug_logger.setLevel(logging.DEBUG)
        
        debug_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        debug_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.experiment_name}_debug_{self.session_id}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=2
        )
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)
    
    # Main logging methods
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
        self.debug_logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    # Experiment-specific logging methods
    def log_experiment_start(self, experiment_type: str, config: Dict[str, Any]):
        """
        Log the start of an experiment
        
        Args:
            experiment_type: Type of experiment (e.g., 'two_players', 'three_players')
            config: Experiment configuration dictionary
        """
        experiment_data = {
            "event": "experiment_start",
            "session_id": self.session_id,
            "experiment_type": experiment_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config
        }
        
        self.exp_logger.info(json.dumps(experiment_data, indent=2))
        self.info(f"🧪 Starting {experiment_type} experiment")
        self.debug(f"Experiment configuration: {json.dumps(config, indent=2)}")
    
    def log_experiment_end(self, experiment_type: str, results: Dict[str, Any], success: bool = True):
        """
        Log the end of an experiment
        
        Args:
            experiment_type: Type of experiment
            results: Experiment results dictionary
            success: Whether the experiment completed successfully
        """
        experiment_data = {
            "event": "experiment_end",
            "session_id": self.session_id,
            "experiment_type": experiment_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": success,
            "results": results
        }
        
        self.exp_logger.info(json.dumps(experiment_data, indent=2))
        
        if success:
            self.info(f"✅ Completed {experiment_type} experiment successfully")
        else:
            self.error(f"❌ Failed {experiment_type} experiment")
        
        self.debug(f"Experiment results: {json.dumps(results, indent=2)}")
    
    def log_algorithm_performance(self, algorithm: str, metrics: Dict[str, Any]):
        """
        Log algorithm performance metrics
        
        Args:
            algorithm: Algorithm name (e.g., 'PPO', 'Gradient')
            metrics: Performance metrics dictionary
        """
        performance_data = {
            "event": "algorithm_performance",
            "session_id": self.session_id,
            "algorithm": algorithm,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.perf_logger.info(json.dumps(performance_data, indent=2))
        
        # Log key metrics to main logger
        if "convergence_time" in metrics:
            self.info(f"⏱️  {algorithm} convergence time: {metrics['convergence_time']:.2f}s")
        if "final_gap" in metrics:
            self.info(f"📊 {algorithm} final gap: {metrics['final_gap']:.3f}")
        if "convergence_quality" in metrics:
            self.info(f"🎯 {algorithm} convergence quality: {metrics['convergence_quality']}")
    
    def log_training_progress(self, algorithm: str, episode: int, metrics: Dict[str, Any]):
        """
        Log training progress for reinforcement learning algorithms
        
        Args:
            algorithm: Algorithm name
            episode: Current episode number
            metrics: Training metrics (effort, reward, loss, etc.)
        """
        progress_data = {
            "event": "training_progress",
            "session_id": self.session_id,
            "algorithm": algorithm,
            "episode": episode,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.debug_logger.debug(json.dumps(progress_data))
        
        # Log periodic progress to main logger
        if episode % 1000 == 0:
            effort = metrics.get("effort", "N/A")
            reward = metrics.get("reward", "N/A")
            self.info(f"📈 {algorithm} Episode {episode}: effort={effort}, reward={reward}")
    
    def log_convergence_check(self, algorithm: str, episode: int, 
                            current_effort: float, theoretical_effort: float,
                            gap: float, std: float, quality: str):
        """
        Log convergence analysis results
        
        Args:
            algorithm: Algorithm name
            episode: Current episode
            current_effort: Current average effort
            theoretical_effort: Theoretical optimal effort
            gap: Gap from theoretical value
            std: Standard deviation of recent efforts
            quality: Convergence quality assessment
        """
        convergence_data = {
            "event": "convergence_check",
            "session_id": self.session_id,
            "algorithm": algorithm,
            "episode": episode,
            "timestamp": datetime.datetime.now().isoformat(),
            "current_effort": current_effort,
            "theoretical_effort": theoretical_effort,
            "gap": gap,
            "std": std,
            "quality": quality
        }
        
        self.debug_logger.debug(json.dumps(convergence_data))
        self.info(f"🔍 {algorithm} Episode {episode}: effort={current_effort:.2f}±{std:.2f}, gap={gap:.3f}, quality={quality}")
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """
        Log error with additional context information
        
        Args:
            error: Exception object
            context: Additional context information
        """
        error_data = {
            "event": "error",
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self.debug_logger.error(json.dumps(error_data, indent=2))
        self.error(f"💥 Error in {context.get('function', 'unknown')}: {str(error)}")
    
    def log_configuration_change(self, component: str, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """
        Log configuration changes
        
        Args:
            component: Component name (e.g., 'PPOAgent', 'Environment')
            old_config: Previous configuration
            new_config: New configuration
        """
        config_data = {
            "event": "configuration_change",
            "session_id": self.session_id,
            "component": component,
            "timestamp": datetime.datetime.now().isoformat(),
            "old_config": old_config,
            "new_config": new_config
        }
        
        self.debug_logger.info(json.dumps(config_data, indent=2))
        self.info(f"⚙️  Configuration changed for {component}")
    
    def close(self):
        """Close all loggers and handlers"""
        self.info(f"🔚 Experiment session ended: {self.session_id}")
        
        # Close all handlers
        for logger in [self.logger, self.exp_logger, self.perf_logger, self.debug_logger]:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

# Global logger instance
_global_logger: Optional[ExperimentLogger] = None

def get_logger(experiment_name: str = "tournament_experiment", **kwargs) -> ExperimentLogger:
    """
    Get or create global logger instance
    
    Args:
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for ExperimentLogger
        
    Returns:
        ExperimentLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(experiment_name, **kwargs)
    return _global_logger

def setup_experiment_logging(experiment_name: str = "tournament_experiment", **kwargs) -> ExperimentLogger:
    """
    Setup experiment logging with custom configuration
    
    Args:
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for ExperimentLogger
        
    Returns:
        ExperimentLogger instance
    """
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
    
    _global_logger = ExperimentLogger(experiment_name, **kwargs)
    return _global_logger

def close_logging():
    """Close global logging"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None

# Convenience functions for quick logging
def log_info(message: str):
    """Quick info logging"""
    logger = get_logger()
    logger.info(message)

def log_debug(message: str):
    """Quick debug logging"""
    logger = get_logger()
    logger.debug(message)

def log_warning(message: str):
    """Quick warning logging"""
    logger = get_logger()
    logger.warning(message)

def log_error(message: str):
    """Quick error logging"""
    logger = get_logger()
    logger.error(message)
