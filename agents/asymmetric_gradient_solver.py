import os

from utils.prob import p_from_efforts
from utils.theory import e_star_two_players_asymmetric_cost

def asymmetric_gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3):
    """
    Gradient descent solver for asymmetric cost parameters.
    Each player optimizes their effort independently given others' efforts.
    
    Args:
        env: AsymmetricCostEnv implementing utility(player_id, effort, *other_efforts)
        lr: learning rate
        steps: number of iterations
        eps: small epsilon for finite-difference gradient
    Returns:
        efforts_final: list of converged effort values for each player
        utilities_final: list of utilities at equilibrium
        costs_final: list of costs at equilibrium
    """
    num_players = env.num_players
    
    # Initialize efforts at midpoint of range
    if hasattr(env, "effort_range"):
        low, high = env.effort_range
        efforts = [(low + high) / 2.0] * num_players
    else:
        efforts = [1.0] * num_players

    log_path = f"/Users/fengjiang/Documents/GSU/tournament_experiment/results/logs/asymmetric_gradient_log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f_log:
        header = "Step," + ",".join([f"Effort{i+1}" for i in range(num_players)]) + "," + \
                ",".join([f"Gradient{i+1}" for i in range(num_players)]) + "," + \
                ",".join([f"Utility{i+1}" for i in range(num_players)]) + "\n"
        f_log.write(header)

    for step in range(steps):
        # Store old efforts for convergence check
        old_efforts = efforts.copy()
        
        # Update each player's effort sequentially (Gauss-Seidel style)
        for player_id in range(num_players):
            # Get other players' current efforts
            other_efforts = [efforts[j] for j in range(num_players) if j != player_id]
            
            # Compute gradient for this player
            u_plus, _ = env.utility(player_id, efforts[player_id] + eps, *other_efforts)
            u_minus, _ = env.utility(player_id, efforts[player_id] - eps, *other_efforts)
            grad = (u_plus - u_minus) / (2 * eps)
            
            # Update this player's effort
            efforts[player_id] += lr * grad
            
            # Clamp to valid range
            if hasattr(env, "effort_range"):
                low, high = env.effort_range
            else:
                low, high = 0.0, 100.0
            efforts[player_id] = min(max(efforts[player_id], low), high)

        # Log current state every 1000 steps
        if step % 1000 == 0:
            with open(log_path, "a") as f_log:
                # Compute current utilities and gradients for logging
                utilities = []
                gradients = []
                for player_id in range(num_players):
                    other_efforts = [efforts[j] for j in range(num_players) if j != player_id]
                    u, _ = env.utility(player_id, efforts[player_id], *other_efforts)
                    utilities.append(u)
                    
                    # Compute gradient for logging
                    u_plus, _ = env.utility(player_id, efforts[player_id] + eps, *other_efforts)
                    u_minus, _ = env.utility(player_id, efforts[player_id] - eps, *other_efforts)
                    grad = (u_plus - u_minus) / (2 * eps)
                    gradients.append(grad)
                
                log_line = f"{step}," + ",".join([f"{e:.6f}" for e in efforts]) + "," + \
                          ",".join([f"{g:.6f}" for g in gradients]) + "," + \
                          ",".join([f"{u:.6f}" for u in utilities]) + "\n"
                f_log.write(log_line)
        
        # Check for convergence
        if step > 1000:
            max_change = max(abs(efforts[i] - old_efforts[i]) for i in range(num_players))
            if max_change < 1e-6:
                print(f"Asymmetric gradient descent converged at step {step}")
                break

    # Compute final utilities and costs
    utilities_final = []
    costs_final = []
    for player_id in range(num_players):
        other_efforts = [efforts[j] for j in range(num_players) if j != player_id]
        u, cost = env.utility(player_id, efforts[player_id], *other_efforts)
        utilities_final.append(u)
        costs_final.append(cost)
    
    return efforts, utilities_final, costs_final

class AsymmetricGradientSolver:
    """Wrapper class for asymmetric gradient descent solver"""
    
    def __init__(self, env, config):
        self.env = env
        self.learning_rate = config.get('learning_rate', 0.01)
        self.max_iterations = config.get('max_iterations', 10000)
    
    def solve(self):
        """Solve using asymmetric gradient descent"""
        efforts, utilities, costs = asymmetric_gradient_descent_solver(
            self.env, 
            lr=self.learning_rate, 
            steps=self.max_iterations
        )
        return efforts, utilities, costs 

def asymmetric_gradient_solver(k1, k2, q, w_h, w_l, effort_range, lr=0.01, max_steps=100000, tol=1e-6):
    """
    优化的不对称梯度下降求解器
    
    Args:
        k1, k2: Cost parameters for players 1 and 2
        q: Noise parameter
        w_h, w_l: High and low rewards
        effort_range: Tuple of (min_effort, max_effort)
        lr: Learning rate
        max_steps: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        e1_final, e2_final, converged, steps
    """
    
    w_gap = w_h - w_l

    def win_prob_and_grad(e1: float, e2: float) -> tuple[float, float]:
        """Return P1's win probability and ∂P1/∂e1 under Uniform(-q, q) noise."""
        diff = e1 - e2
        if diff <= -2.0 * q:
            return 0.0, 0.0
        if diff >= 2.0 * q:
            return 1.0, 0.0
        prob = float(p_from_efforts(e1, e2, q))
        grad = (1.0 / (2.0 * q)) - (abs(diff) / (4.0 * q * q))
        return prob, grad

    def analytical_gradient1(e1: float, e2: float) -> float:
        _, dp_de1 = win_prob_and_grad(e1, e2)
        return w_gap * dp_de1 - 2.0 * k1 * e1

    def analytical_gradient2(e1: float, e2: float) -> float:
        _, dp_de1 = win_prob_and_grad(e1, e2)
        return w_gap * dp_de1 - 2.0 * k2 * e2

    # 使用闭式解作为智能初始化
    min_effort, max_effort = effort_range
    e1_star, e2_star = e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)

    def clip(value: float) -> float:
        return max(min_effort, min(max_effort, value))

    e1 = clip(e1_star)
    e2 = clip(e2_star)

    print(f"🎯 智能初始化: e1={e1:.3f}, e2={e2:.3f} (闭式解)")
    
    # 自适应学习率
    lr_initial = lr
    lr_current = lr_initial
    momentum1, momentum2 = 0.0, 0.0
    momentum_decay = 0.9
    
    best_gap = float('inf')
    patience = 0
    max_patience = 5000
    
    for step in range(max_steps):
        old_e1, old_e2 = e1, e2
        
        # 计算解析梯度
        grad1 = analytical_gradient1(e1, e2)
        grad2 = analytical_gradient2(e1, e2)
        
        # 添加动量
        momentum1 = momentum_decay * momentum1 + (1 - momentum_decay) * grad1
        momentum2 = momentum_decay * momentum2 + (1 - momentum_decay) * grad2
        
        # 更新参数
        new_e1 = e1 + lr_current * momentum1
        new_e2 = e2 + lr_current * momentum2
        
        # 限制在有效范围内
        new_e1 = max(min_effort, min(max_effort, new_e1))
        new_e2 = max(min_effort, min(max_effort, new_e2))
        
        e1, e2 = new_e1, new_e2
        
        # 检查收敛
        change1 = abs(e1 - old_e1)
        change2 = abs(e2 - old_e2)
        max_change = max(change1, change2)
        
        # 计算与理论值的gap (用于监控)
        gap1 = abs(e1 - e1_star)
        gap2 = abs(e2 - e2_star)
        current_gap = (gap1 + gap2) / 2
        
        if current_gap < best_gap:
            best_gap = current_gap
            patience = 0
        else:
            patience += 1
        
        # 自适应学习率调整
        if step > 1000 and step % 500 == 0:
            if patience > 200:
                lr_current *= 0.95  # 降低学习率
                patience = 0
            elif max_change < tol * 10:
                lr_current *= 1.05  # 稍微提高学习率
                lr_current = min(lr_current, lr_initial * 2)
        
        # 收敛检查
        if max_change < tol:
            print(f"✅ 收敛于步骤 {step}: e1={e1:.6f}, e2={e2:.6f}")
            return e1, e2, True, step + 1
        
        # 早停检查
        if patience > max_patience:
            print(f"⏹️ 早停于步骤 {step}: gap未改善超过{max_patience}步")
            return e1, e2, False, step + 1
        
        # 打印进度
        if step % 5000 == 0:
            print(f"步骤 {step}: e1={e1:.6f}, e2={e2:.6f}, 变化=({change1:.8f}, {change2:.8f}), gap={current_gap:.6f}, lr={lr_current:.6f}")
    
    # 未收敛
    print(f"⚠️ 未收敛: 达到最大步数 {max_steps}")
    return e1, e2, False, max_steps 
