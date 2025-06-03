import torch

# -----------------------------------------------------
# 1) Random initialization of efforts in [0, 100]
# -----------------------------------------------------
torch.manual_seed(42)

player1_effort = (torch.rand(1) * 100).requires_grad_(True)
player2_effort = (torch.rand(1) * 100).requires_grad_(True)

print(f"Initial effort of Player 1: {player1_effort.item():.2f}")
print(f"Initial effort of Player 2: {player2_effort.item():.2f}")

# -----------------------------------------------------
# 2) Tournament parameters
# -----------------------------------------------------
w_H = 6.0   # Winning prize
w_L = 2.0   # Losing (base) prize
q   = 50.0  # Noise range parameter (ε ~ Uniform(-q, q))
k   = 1/3500  # Cost coefficient (cost = k * e^2)

# -----------------------------------------------------
# 3) Exact probability function for y1 > y2
#    where y1 = e1 + eps1, y2 = e2 + eps2, eps ~ U(-q, q)
# -----------------------------------------------------
def probability_of_winning(e1, e2, q):
    d = e2 - e1  # We want P(D > d), where D = eps1 - eps2
    d_clamped = torch.clamp(d, -2*q, 2*q)
    
    p_neg = 1.0 - (d_clamped + 2*q)**2 / (8*q*q)   # if d_clamped < 0
    p_pos = (2*q - d_clamped)**2 / (8*q*q)         # if d_clamped >= 0

    mask_neg = (d_clamped < 0).float()
    p_middle = mask_neg * p_neg + (1 - mask_neg) * p_pos

    p_final = torch.where(d < -2*q,
                          torch.tensor(1.0, dtype=d.dtype, device=d.device),
                          torch.where(d > 2*q,
                                      torch.tensor(0.0, dtype=d.dtype, device=d.device),
                                      p_middle))
    return p_final

# -----------------------------------------------------
# 4) Utility functions using the exact probability
# -----------------------------------------------------
def u1(e1, e2):
    p = probability_of_winning(e1, e2, q)
    return w_L + p*(w_H - w_L) - k * e1**2

def u2(e1, e2):
    p = probability_of_winning(e1, e2, q)
    return w_L + (1 - p)*(w_H - w_L) - k * e2**2

# -----------------------------------------------------
# 5) Optimizer & custom loss
#    Removed the term -λ*(e1 + e2)
# -----------------------------------------------------
optimizer = torch.optim.Adam([player1_effort, player2_effort], lr=0.01)

for step in range(20000):
    optimizer.zero_grad()

    # Compute current utilities
    u1_val = u1(player1_effort, player2_effort)
    u2_val = u2(player1_effort, player2_effort)

    # Gradients wrt each player's own effort
    grad_u1 = torch.autograd.grad(u1_val, player1_effort, create_graph=True)[0]
    grad_u2 = torch.autograd.grad(u2_val, player2_effort, create_graph=True)[0]

    # Loss = (∂u1/∂e1)² + (∂u2/∂e2)²
    loss = grad_u1**2 + grad_u2**2
    loss.backward()
    optimizer.step()

    # Clamp efforts to be nonnegative
    with torch.no_grad():
        player1_effort.clamp_(min=0)
        player2_effort.clamp_(min=0)

    # Print progress every 200 steps
    if step % 200 == 0:
        print(f"Step {step:4d}: e1={player1_effort.item():.2f}, "
              f"e2={player2_effort.item():.2f}, loss={loss.item():.4f}")

# -----------------------------------------------------
# 6) Final results
# -----------------------------------------------------
# Theoretical symmetric equilibrium effort: e* = (wH - wL) / (4*q*k)
e_star = (w_H - w_L) / (4 * q * k)
print(f"\nTheoretical symmetric equilibrium effort: {e_star:.2f}")
print("\nFinal results:")
print(f"Player 1 effort: {player1_effort.item():.2f}")
print(f"Player 2 effort: {player2_effort.item():.2f}")