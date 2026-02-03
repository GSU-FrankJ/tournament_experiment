
# Experiment Plan for Exploring the Tournament Equilibrium

## I. Methodology

### 1. Gradient-based Optimization

- Directly optimize the efforts of player 1 and player 2 using gradient-based methods.
- A benchmark solution exists in closed form, which is suitable for evaluating numerical convergence.
- **Limitations**: This approach only works for static, symmetric, single-round games and does **not** generalize to:
  - Multi-stage interactions
  - More than two players
  - Additional randomness
  - Learning from trajectories (state-action sequences)

### 2. Reinforcement Learning with Self-play

We adopt a reinforcement learning (RL) framework where both agents are trained via self-play using policy-based methods. Two techniques are considered:

#### a. Policy Gradient - REINFORCE

- **Objective function**:  
  \( J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right] \)  
  Where:  
  - \( \theta \): parameters of the policy network  
  - \( \tau = (s_1,a_1, \ldots, s_T, a_T) \): trajectory  
  - \( R(\tau) \): total reward along the trajectory  

- **Gradient**:  
  \( \nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right] \)

#### b. Proximal Policy Optimization (PPO)

- **Clipped surrogate objective**: prevents large updates to improve training stability and sample efficiency.  
  \[
  L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
  \]  
  Where:  
  - \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \): probability ratio  
  - \( \hat{A}_t \): estimated advantage  
  - \( \epsilon \): clipping threshold

## II. Training with Different Parameters

1. **Utility Weights**: \( w_H = 6.5, w_L = 3 \)  
2. **Parameter**: \( k = \frac{1}{2500} = 0.0004 \)  
3. **Effort Ranges**: \([0, 100]\), \([0, 200]\)  
4. **Noise Term**: \( \epsilon \sim \text{Uniform}(-q, q) \) with different \( q \)

## III. Three Groups of Experiments

### 1. One-stage Tournament with Two Identical Players

- **Utility function**: \( U = V - C \)
- **Stage output**: \( V = e_1 + e_2 \), with \( e_1 = e_2 = e \)
- **Cost function**: \( C(e) = ke^2 \)
- **Expected utility**:  
  \( \mathbb{E}[U] = \frac{w_H + w_L}{2} - ke^2 \)
- **Symmetric equilibrium effort**: closed-form solution exists

### 2. One-stage Tournament Expansion

#### a. Three Identical Competitors

- One winner, two losers
- Output: \( V = e_1 + e_2 + e_3 \)
- Cost: \( C(e) = ke^2 \)
- Utility structure updated accordingly
- Symmetric equilibrium effort derived

#### b. Two Players with Different Cost Functions

- Cost: \( C_i(e) = k_i e^2 \)
- Output: \( V = e_1 + e_2 \)
- Payoffs, utility functions differ per player
- Equilibrium effort derived per player

#### c. Players with Different Abilities

- \( V = \alpha_i e_i + e_j \) with ability factor \( \alpha_i \)
- Payoff depends on ability-weighted effort
- Cost: \( C(e) = ke^2 \)
- Symmetric and asymmetric effort analysis

### 3. Two-stage Tournament

- **Utility**: \( U = V - C \)
- **Stage output**: \( V_t = e_{1t} + e_{2t} \)
- **Cost**: \( C_t(e) = ke^2 \)
- **Expected payoff**:  
  \( \mathbb{E}[U] = \sum_{t=1}^2 \left( \mathbb{E}[V_t] - C_t \right) \)
- **Symmetric equilibrium**:
  - Stage 1 effort: \( e^*_1 \)
  - Stage 2 effort: \( e^*_2 \)

## IV. Experiment Analysis

### 1. Equilibrium Verification

- Check whether learned strategies (from Gradient, REINFORCE, PPO) converge to closed-form Nash equilibrium under different settings.

### 2. Generalization of RL Models

- Test how well RL methods adapt to changes in:
  - Number of players
  - Cost structures
  - Tournament formats

### 3. Stability and Convergence Comparison

- Compare across:
  - Gradient-based optimization
  - REINFORCE
  - PPO  
- Metrics: effort stability, convergence speed, deviation from theoretical optimum
