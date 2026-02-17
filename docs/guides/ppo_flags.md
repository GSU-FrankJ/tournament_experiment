# PPO Training Flags Reference

Three key flags control the modern PPO training pipeline. All three are enabled
by default when running with `--method ppo`.

```bash
python run/run_two_players.py --method ppo --q 25 --seed 42
# Equivalent to:
#   --theory-align-v2 --enable-convergence-eval --cheap-gate-profile relaxed
```

---

## `--theory-align-v2`

**What it does:** Swaps the policy network architecture.

| | Default (`ActorCritic`) | `--theory-align-v2` (`ActorCriticMeanConc`) |
|---|---|---|
| Network outputs | raw alpha, beta | **mean** (sigmoid) + **concentration** (softplus) |
| Action distribution | `Beta(alpha, beta)` | `Beta(mean * conc, (1 - mean) * conc)` |
| Interpretation | alpha/beta have no direct meaning | mean = expected effort, conc = certainty |

**Why it helps:** The mean+concentration parameterization decouples *where to aim*
(mean) from *how sure the agent is* (concentration). This makes the learned policy
directly interpretable — the mean output is comparable to the theoretical
equilibrium effort e\* — and tends to train more stably than raw alpha/beta.

**Does it require a closed-form equilibrium?** No. PPO learns via self-play
regardless. The closed-form e\* is only needed for two optional auxiliary losses
(both off by default):

| Optional feature | Needs e\*? | Default | Flag |
|---|---|---|---|
| Variance penalty | No | `0.0` | `--theory-align-v2-var-coef` |
| Best-response penalty `(mean - e*)^2` | Yes | `0.0` | `--theory-align-v2-br-coef` |
| Init bias near e\* | Yes | `None` | `--init-bias-mean` |

---

## `--enable-convergence-eval`

**What it does:** Enables automatic early stopping based on exploitability.

Without this flag, training runs for a fixed number of episodes. With it, the
system periodically evaluates whether the learned policy is an approximate Nash
equilibrium and stops early when it is.

### Convergence criterion

A policy has converged when **exploitability < exploit\_eps** for
**patience\_exploit** consecutive evaluations.

- `exploit_eps = 0.03` (default) — the epsilon in epsilon-Nash
- `patience_exploit = 5` (default) — consecutive passes required

### Exploitability evaluation

At each check, the system computes a best-response via coarse-to-fine grid
search with Monte Carlo sampling (`M = 8192` samples). If a unilateral deviation
can improve payoff by more than `exploit_eps`, the policy is not yet at
equilibrium.

### Two-level gating

Exploitability evaluation is expensive. To avoid running it every update, two
mechanisms control when it fires:

1. **Cheap gate** (see below) — fast stability check that triggers eval early
   when the policy looks stable
2. **Periodic fallback** (`exploit_every_updates = 10`) — guarantees eval runs at
   least every N updates, regardless of the cheap gate

### Output

When convergence is detected, training stops and the convergence JSON records:
- `stop_reason: "exploitability"` or `"max_updates"`
- `stopped_at_update`, `final_exploit_1`, `final_exploit_2`
- Full history of effort, KL, and exploitability values per update

---

## `--cheap-gate-profile <profile>`

**What it does:** Configures the cheap gate — a fast pre-filter that controls
*when* the expensive exploitability evaluation runs.

The cheap gate monitors three rolling-window statistics and only triggers an
exploit eval when all three are stable:

| Metric | What it measures |
|---|---|
| `mean_kl_window` | Average KL divergence over the last `window_size` updates |
| `std_kl_window` | Volatility of KL divergence (standard deviation) |
| `drift_effort` | Absolute change in mean effort from window start to end |

All three must pass their thresholds for `patience_drift` consecutive checks
before the gate triggers.

### Available profiles

| Profile | mean\_kl\_thresh | std\_kl\_thresh | drift\_effort\_thresh | patience\_drift | Use case |
|---|---|---|---|---|---|
| **relaxed** | 0.015 | 0.012 | 8.0 | 1 | Default for PPO with theory-align-v2 |
| default | 0.0045 | 0.0035 | 2.0 | 2 | Original baseline |
| conservative | 0.0038 | 0.0030 | 1.5 | 3 | Strictest — longest stability proof |
| aggressive | 0.0060 | 0.0075 | 5.5 | 1 | Loosest — fastest trigger |

### Why "relaxed" is the default

The mean+concentration parameterization from `--theory-align-v2` produces
different KL divergence patterns than raw alpha/beta. KL fluctuations are
naturally larger, so the original thresholds (0.0045) would almost never pass,
making the gate useless. The relaxed profile accommodates this.

### Key insight: the gate is a cost optimization, not a convergence criterion

The cheap gate only controls *when* exploit eval runs — it does **not** affect
the convergence criterion itself (which is always exploitability < exploit\_eps).
Even if the gate never triggers, the periodic fallback ensures exploit eval runs
every `exploit_every_updates` steps. This means:

- A too-tight gate: wastes time by delaying exploit checks, but same final result
- A too-loose gate: runs exploit checks more often, wastes compute, but same final result
- `--disable-cheap-gate`: gate always passes, exploit eval eligible every update

The gate parameters do **not** require ablation for paper validity.

---

## Interaction diagram

```
Every eval_every_updates steps:
  |
  +--> Compute KL + drift over rolling window
  |
  +--> Cheap gate passes?
  |      |
  |      +-- YES (streak >= patience_drift) --> eligible for exploit eval
  |      +-- NO  --> wait for periodic fallback
  |
  +--> Run exploitability evaluation?
  |      (triggered by gate OR periodic interval)
  |
  +--> exploitability < exploit_eps?
  |      |
  |      +-- YES --> increment exploit_ok_streak
  |      +-- NO  --> reset exploit_ok_streak (and drift_ok_streak)
  |
  +--> exploit_ok_streak >= patience_exploit?
         |
         +-- YES --> STOP TRAINING (converged)
         +-- NO  --> continue
```
