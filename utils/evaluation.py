#!/usr/bin/env python3
"""
Evaluation Utilities for Tournament Experiments
==============================================

This module provides evaluation functions for assessing algorithm performance
according to experimental optimization standards.
"""

def calculate_quality(gap: float) -> str:
    """
    Calculate performance quality based on gap from theoretical value
    
    Performance Standards:
    - Excellent: Gap < 0.5 (与理论值差距小于0.5)
    - Good: Gap < 1.0 (与理论值差距小于1.0)  
    - Fair: Gap < 5.0 (与理论值差距小于5.0)
    - Poor: Gap >= 5.0 (不可接受)
    
    Args:
        gap: Absolute difference from theoretical value
        
    Returns:
        Quality string: "Excellent", "Good", "Fair", or "Poor"
    """
    if gap < 0.5:
        return "Excellent"
    elif gap < 1.0:
        return "Good"
    elif gap < 5.0:
        return "Fair"
    else:
        return "Poor"

def validate_performance_standard(gap: float, algorithm: str = "Algorithm") -> bool:
    """
    Validate if performance meets minimum standards
    
    Args:
        gap: Gap from theoretical value
        algorithm: Algorithm name for error reporting
        
    Returns:
        True if meets standard (Good or better), False otherwise
    """
    quality = calculate_quality(gap)
    meets_standard = quality in ["Excellent", "Good"]
    
    if not meets_standard:
        print(f"❌ {algorithm} performance below standard: {quality} (gap={gap:.3f})")
    else:
        print(f"✅ {algorithm} meets performance standard: {quality} (gap={gap:.3f})")
    
    return meets_standard

def calculate_convergence_score(episodes: int, max_episodes: int = 20000) -> float:
    """
    Calculate convergence efficiency score based on training episodes
    
    Args:
        episodes: Number of episodes used for convergence
        max_episodes: Maximum possible episodes
        
    Returns:
        Score from 0.0 to 1.0 (higher is better)
    """
    if episodes >= max_episodes:
        return 0.0
    
    # Linear scoring: fewer episodes = higher score
    return 1.0 - (episodes / max_episodes)

def summarize_experiment_results(results: list) -> dict:
    """
    Summarize results from multiple experiments
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        Summary statistics dictionary
    """
    if not results:
        return {}
    
    total_tests = len(results)
    excellent_count = sum(1 for r in results if r.get("quality") == "Excellent")
    good_count = sum(1 for r in results if r.get("quality") == "Good")
    fair_count = sum(1 for r in results if r.get("quality") == "Fair")
    poor_count = sum(1 for r in results if r.get("quality") == "Poor")
    
    success_rate = (excellent_count + good_count) / total_tests * 100
    
    avg_gap = sum(r.get("avg_gap", 0) for r in results) / total_tests
    
    summary = {
        "total_tests": total_tests,
        "excellent_count": excellent_count,
        "good_count": good_count,
        "fair_count": fair_count,
        "poor_count": poor_count,
        "success_rate": success_rate,
        "average_gap": avg_gap,
        "meets_standard": success_rate >= 100.0
    }
    
    return summary

def print_performance_matrix(results: list, algorithms: list, test_conditions: list):
    """
    Print a formatted performance matrix for multiple algorithms and conditions
    
    Args:
        results: List of experiment results
        algorithms: List of algorithm names
        test_conditions: List of test condition identifiers
    """
    print("\n📈 性能矩阵:")
    
    # Create header
    header = "算法".ljust(12)
    for condition in test_conditions:
        header += f" | {condition}".ljust(12)
    print(header)
    print("-" * len(header))
    
    # Create matrix
    for algorithm in algorithms:
        row = algorithm.ljust(12)
        for condition in test_conditions:
            # Find result for this algorithm and condition
            result = None
            for r in results:
                if (r.get("Model_training") == algorithm and 
                    str(r.get("test_condition", "")).startswith(str(condition))):
                    result = r
                    break
            
            if result:
                quality = result.get("quality", "Unknown")
                gap = result.get("avg_gap", 0)
                if quality == "Excellent":
                    symbol = "★"
                elif quality == "Good":
                    symbol = "✓"
                elif quality == "Fair":
                    symbol = "○"
                else:
                    symbol = "✗"
                cell = f" | {symbol}  {gap:.1f}".ljust(12)
            else:
                cell = " | N/A".ljust(12)
            row += cell
        print(row)
