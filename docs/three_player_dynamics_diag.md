# Three-Player Training Dynamics Diagnosis

Data sources:
- 3P: `results/three_players/convergence/ppo_3p_q35.0_seed42_baseline_convergence.json` (warmup=200 sanity run, 44 updates)
- 2P: `results/two_players/convergence/ppo_q35.0_seed42_convergence.json` (Round 2, 69 updates)

## 1. Effort trajectory

### 3P (q=35, seed=42, e\*=25.00)

| metric | value |
|--------|------:|
| n_updates | 44 |
| initial (upd 0) | 50.20 |
| minimum (upd 36) | 20.57 |
| final (upd 43) | 21.88 |
| final gap | 3.12 (12.47%) |
| max descent rate | -1.27 (upd 13→14) |
| max ascent rate | +0.41 (upd 42→43) |

Checkpoints:

| upd | effort |
|----:|-------:|
| 0 | 50.20 |
| 11 | 38.57 |
| 22 | 26.12 |
| 33 | 20.74 |
| 36 | 20.57 |
| 43 | 21.88 |

Effort overshoots past e\*=25.00 around update 23, reaches minimum at update 36 (20.57, 4.43 units below e\*), then partially recovers to 21.88 before early stop.

### 2P (q=35, seed=42, e\*=45.45)

| metric | value |
|--------|------:|
| n_updates | 69 |
| initial (upd 0) | 57.90 |
| minimum (upd 61) | 42.67 |
| final (upd 68) | 43.46 |
| final gap | 2.00 (4.40%) |
| max descent rate | -0.57 (upd 6→7) |
| max ascent rate | +0.60 (upd 66→67) |

Checkpoints:

| upd | effort |
|----:|-------:|
| 0 | 57.90 |
| 17 | 49.94 |
| 34 | 45.03 |
| 51 | 43.82 |
| 68 | 43.46 |

2P also overshoots below e\*, but the descent rate is slower (max -0.57 vs -1.27) and the overshoot is smaller relative to e\*.

## 2. Exploitability trajectory

### 3P

| upd | exploitability |
|----:|---------------:|
| 0 | 1.3788 |
| 10 | 0.5624 |
| 20 | 0.0595 |
| 30 | 0.0160 |
| 39 | 0.0165 |
| 40 | 0.0215 |
| 41 | 0.0177 |
| 42 | 0.0217 |
| 43 | 0.0174 |

Exploit OK streak of 5 reached at **update 42** (started at update 30).

Best-response efforts at final eval: `final_br_effort_1 = 25.75`, `final_br_effort_2 = 25.75`.

Policy is at 21.88 but BR says 25.75. The gap (21.88 vs 25.75 = 3.87 units) is large, yet exploitability is only 0.0174. This means the EU gain from deviating from 21.88 to 25.75 is small in absolute terms.

### 2P

| upd | exploitability |
|----:|---------------:|
| 8 | 0.1341 |
| 18 | 0.0518 |
| 28 | 0.0155 |
| 38 | 0.0151 |
| 48 | 0.0073 |
| 58 | 0.0067 |
| 68 | 0.0072 |

Exploit OK streak of 5 reached at **update 68** (started at update 28).

Best-response efforts: `final_br_effort_1 = 43.0`, `final_br_effort_2 = 43.0`.

## 3. Concentration trajectory

### 3P

`alpha_mean` and `beta_mean`: **MISSING** from convergence JSON.

### 2P

| upd | alpha+beta |
|----:|----------:|
| 0 | 162.1 |
| 10 | 160.7 |
| 20 | 159.3 |
| 30 | 163.8 |
| 40 | 170.0 |
| 68 | 195.9 |

alpha: 93.84 → 85.14; beta: 68.24 → 110.79.

## 4. KL / Entropy

| metric | 3P | 2P |
|--------|---:|---:|
| approx_kl first 10 mean | 0.013116 | 0.002442 |
| approx_kl last 10 mean | 0.004216 | 0.002291 |
| batch_entropy start | -1.8116 | -1.8354 |
| batch_entropy end | -2.2536 | -1.9233 |

3P has ~5x higher KL in the first 10 updates. Entropy drops more aggressively in 3P (-1.81 → -2.25) vs 2P (-1.84 → -1.92).

## 5. Gradient signal (dEU/de at policy_mean)

Formula: `dEU/de = W/(2q) - 2k*e` where W=3.5, q=35.

At e\*: `dEU/de = 0.05 - 2k*e* = 0`.

### 3P (k=0.001)

| upd | effort | dEU/de | direction |
|----:|-------:|-------:|-----------|
| 0 | 50.20 | -0.0504 | toward e\* |
| 4 | 46.26 | -0.0425 | toward e\* |
| 8 | 41.94 | -0.0339 | toward e\* |
| 12 | 37.39 | -0.0248 | toward e\* |
| 16 | 32.69 | -0.0154 | toward e\* |
| 20 | 28.22 | -0.0064 | toward e\* |
| 24 | 24.30 | +0.0014 | toward e\* |
| 28 | 22.08 | +0.0058 | toward e\* |
| 32 | 20.70 | +0.0086 | toward e\* |
| 36 | 20.57 | +0.0089 | toward e\* |
| 40 | 21.06 | +0.0079 | toward e\* |
| 43 | 21.88 | +0.0062 | toward e\* |

Gradient is always pointing toward e\*. After overshoot (upd ~24), the gradient switches sign to positive (pushing effort up toward 25), but the magnitude is small: 0.006-0.009. The ascent rate (+0.41/update max) is much slower than the descent rate (-1.27/update max).

### 2P (k=0.00055)

| upd | effort | dEU/de | direction |
|----:|-------:|-------:|-----------|
| 0 | 57.90 | -0.0137 | toward e\* |
| 12 | 51.76 | -0.0069 | toward e\* |
| 24 | 48.26 | -0.0031 | toward e\* |
| 36 | 45.58 | -0.0001 | toward e\* |
| 48 | 43.71 | +0.0019 | toward e\* |
| 60 | 42.86 | +0.0029 | toward e\* |
| 68 | 43.46 | +0.0022 | toward e\* |

### Gradient magnitude comparison at equal distance from e\*

At 5 units below e\* (e = e\* - 5):
- 3P: dEU/de = 0.05 - 2(0.001)(20) = **+0.010**
- 2P: dEU/de = 0.05 - 2(0.00055)(40.45) = **+0.0055**

3P has a ~1.8x stronger gradient at the same absolute distance from e\*. Despite this, the policy stalls — the ascent rate is only +0.41/update at best vs the descent rate of -1.27.

## 6. Side-by-side summary

| metric | 3P | 2P |
|--------|---:|---:|
| e\* | 25.00 | 45.45 |
| k | 0.001 | 0.00055 |
| n_updates | 44 | 69 |
| initial effort | 50.20 | 57.90 |
| overshoot depth (below e\*) | 4.43 | 2.78 |
| final effort | 21.88 | 43.46 |
| final gap (abs) | 3.12 | 2.00 |
| final gap (rel%) | 12.47% | 4.40% |
| exploit streak start | upd 30 | upd 28 |
| exploit streak end | upd 42 | upd 68 |
| updates after streak start | 14 | 42 |
| BR effort at stop | 25.75 | 43.0 |
| KL first 10 mean | 0.0131 | 0.0024 |
| entropy drop | 0.44 | 0.09 |
| concentration at upd 40 | MISSING | 170.0 |
