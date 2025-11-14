# One-Stage Two-Player PPO Sweeps Summary

## Title & Context
This document summarizes every one-stage two-player PPO sweep currently stored in this repository. All numbers come from the latest implementation of `run/run_two_players.py` and `config/one_stage_two_players.py`, plus the sweep scripts and CSVs under `run/` and `results/`. The goal is to record how the experimentation pathway moved from conservative settings toward more aggressive ones in pursuit of the theoretical equilibrium effort at q = 55.

## Baseline Setup (from current code)
- **Environment:** `run/run_two_players.py` instantiates `TwoPlayersEnv` for a one-stage, two-player contest; the learned stage is logged as stage 2 while stage-1 columns are fixed at 0. Effort bounds are `[0, 200]` for stage 2 and `[0, 100]` for stage 1 (`config/one_stage_two_players.py`).
- **Theoretical reference:** `utils/theory.py` defines `e_star_two_players(q, w_h, w_l, k) = (w_H - w_L) / (4 q k)`. With `w_h = 6.5`, `w_l = 3.0`, `k = 0.0004`, and `q = 55`, the CSVs report the theoretical target `39.772727`.
- **PPO architecture:** `run/run_two_players.py` builds `PPOTwoPlayersBandit` (from `agents/ppo_two_players_clean.py`) with a Beta policy head, shared MLP hidden size 128, and a 3-D state vector `(q, k, w_gap)`.
- **Default training schedule (per `config/one_stage_two_players.py`):**
  - `steps_per_update = 4096`, `update_epochs = 6`, `minibatch_size = 1024`.
  - `episodes = 2_048_000` with `max_updates = 500`, so PPO runs for roughly `2_048_000 / 4096 = 500` updates even if a sweep bumps the requested episode count higher.
  - Opponent / lag: `opponent_mode = "periodic"`, `opponent_sync_interval = 2`, `opponent_ema_tau = 0.20`, `opponent_snapshot_keep = 10`, lag warmup of 10 updates followed by a 10-update linear fade, and `opponent_history_sample_p` held at 0.
  - Schedules: entropy coefficient starts and holds at `0.02` before annealing to `0.008`; learning rate goes from `3e-4` toward `2e-4`; clip range ramps from `0.25` down to `0.18` with floor `0.10` and ceiling `0.60`.
  - KL controller: `target_kl = 0.015`, `kl_low = 0.5 * target`, `kl_high = 3 * target`, with clip-factor multipliers of `1.5`/`0.7` and LR multipliers of `1.5`/`0.7`; `warm_decay_ratio = 0.7` and `force_kl_gate = True` gate schedule decay until KL clears the low threshold.

## Sweep Overview (Timeline)
### Original Sweep
**Code location**
- `run/hparam_sweep_two_players.py`
- `results/one_stage_two_players_sweep_q55.csv`

**Hyperparameter design**
| ID | clip_start → clip_end | entropy_start/hold | entropy_end | target_kl |
| --- | --- | --- | --- | --- |
| A_baseline_soft | 0.25 → 0.16 | 0.02 | 0.004 | 0.010 |
| B_more_explore | 0.25 → 0.18 | 0.03 | 0.006 | 0.015 |
| C_conservative | 0.22 → 0.14 | 0.02 | 0.002 | 0.010 |
| D_clip_wide | 0.25 → 0.18 | 0.02 | 0.005 | 0.010 |

**Training budget**
- Episodes inherit the config value `2,048,000`. With `steps_per_update = 4,096`, this is `2,048,000 / 4,096 = 500` PPO updates, aligned with `max_updates`.

**Results summary (q = 55, seed = 42)**
- 4 PPO rows in `results/one_stage_two_players_sweep_q55.csv`; ordering matches the experiment list, so the best row corresponds to `A_baseline_soft`.
- Best metrics: `final_stage2_effort = 38.0776` vs theoretical `39.7727` (`abs_err = 1.6951`, `Gap_from_theoretical = 11.0182`). `approx_kl = 1.63e-4`, `batch_entropy = -2.5165`, `alpha_mean = 76.39`, `beta_mean = 324.85`. Opponent settings logged as `periodic`, sync interval `2`, `opp_ema_tau = 0.2`, and `last_sync_step = 500`.
- Key hyperparameters for this row are the script defaults for `A_baseline_soft`: clip `0.25 → 0.16`, entropy end `0.004`, `target_kl = 0.010`, with lag warmup/fade `10/10` from the shared config.

**Qualitative note**
- All four runs undershot the 39.77 target by 1.7–2.36 units despite the different entropy tails.
- KL never left the `1e-4` range, so the adaptive clip/LR logic never tightened around the `0.010` target.

### Aggressive Sweep
**Code location**
- `run/hparam_sweep_two_players_aggressive.py`
- `results/one_stage_two_players_sweep_q55_aggressive.csv`

**Hyperparameter design**
| ID | clip_start → clip_end | entropy_start/hold | entropy_end | target_kl |
| --- | --- | --- | --- | --- |
| A_mild | 0.25 → 0.18 | 0.02 | 0.005 | 0.015 |
| B_kl_high | 0.25 → 0.20 | 0.02 | 0.005 | 0.020 |
| C_explore_more | 0.25 → 0.20 | 0.03 | 0.006 | 0.015 |
| D_very_aggressive | 0.30 → 0.22 | 0.03 | 0.008 | 0.025 |
| E_wide_clip_safe_kl | 0.28 → 0.20 | 0.02 | 0.005 | 0.015 |
| F_baseline_plus | 0.25 → 0.16 | 0.02 | 0.004 | 0.012 |

**Training budget**
- Episodes were set to `3,000,000`, which would be `≈732` updates at 4,096 steps/update, but `max_updates = 500` still caps execution at 500 updates (`≈2.048M` steps).

**Results summary (q = 55, seed = 42)**
- 6 rows logged; the fourth row (`D_very_aggressive`) is best with `final_stage2_effort = 38.5644` vs `39.7727` (`abs_err = 1.2083`, `Gap_from_theoretical = 7.8539`).
- Diagnostics: `approx_kl = 5.85e-4`, `batch_entropy = -2.4966`, `alpha_mean = 74.72`, `beta_mean = 312.80`; opponents remain `periodic` with sync `2`, `opp_ema_tau = 0.2`, and `last_sync_step = 500`.
- Sweep-specific knobs for this winner: clip `0.30 → 0.22`, entropy tail `0.008`, and `target_kl = 0.025` (with KL low/high inherited as `0.0125/0.075`).

**Qualitative note**
- Wide clips helped shrink the gap to ~1.2, but the logged KL stayed 20× below the 0.025 target, so the controller never left its most permissive regime.
- Entropy settled a bit higher (-2.50 vs -2.52) but Beta heads remained around 75/313, indicating only a modest move toward deterministic actions.

### Sweep0
**Code location**
- `run/hparam_sweep_two_players_sweep0.py`
- `results/one_stage_two_players_sweep0_q55.csv`

**Hyperparameter design**
| ID | clip_start → clip_end | entropy_start/hold | entropy_end | target_kl |
| --- | --- | --- | --- | --- |
| sweep0_A_mild | 0.30 → 0.22 | 0.02 | 0.006 | 0.020 |
| sweep0_B_kl_high | 0.32 → 0.24 | 0.02 | 0.006 | 0.025 |
| sweep0_C_explore_more | 0.30 → 0.22 | 0.03 | 0.008 | 0.020 |
| sweep0_D_very_aggressive | 0.35 → 0.26 | 0.03 | 0.010 | 0.030 |
| sweep0_E_wide_clip_safe | 0.30 → 0.22 | 0.025 | 0.007 | 0.018 |
| sweep0_F_control | 0.25 → 0.18 | 0.02 | 0.005 | 0.015 |

**Training budget**
- Requested episodes were again `3,000,000`, but `max_updates = 500` limits the actual PPO updates to 500 (`2.048M` steps).

**Results summary (q = 55, seed = 42)**
- 6 runs; the third row (`sweep0_C_explore_more`) minimized `abs_err` at `1.1359`. It reached `final_stage2_effort = 38.6369`, so the remaining gap to theory was `7.3831`.
- Logged stats: `approx_kl = -2.28e-4` (numerical noise near zero), `batch_entropy = -2.4906`, `alpha_mean = 74.49`, `beta_mean = 311.10`, `opp_mode = periodic`, `opp_sync_interval = 2`, `opp_ema_tau = 0.2`, `last_sync_step = 500`.
- The winning config used clip `0.30 → 0.22`, entropy tail `0.008`, and `target_kl = 0.020`.

**Qualitative note**
- Even the aggressive KL=0.020 setting left KL near zero, so tightening never kicked in; the improvement over the aggressive sweep is mainly from the entropy boost.
- Alpha/Beta means shifted only slightly (74/311), suggesting the policy remains relatively diffuse despite the wider clip band.

### Sweep1
**Code location**
- `run/hparam_sweep_two_players_sweep1.py`
- `results/one_stage_two_players_sweep1_q55.csv`

**Hyperparameter design**
| ID | clip_start → clip_end | entropy_start/hold | entropy_end | target_kl |
| --- | --- | --- | --- | --- |
| sweep1_A_kl0.05 | 0.35 → 0.25 | 0.02 | 0.008 | 0.05 |
| sweep1_B_kl0.06 | 0.40 → 0.28 | 0.02 | 0.008 | 0.06 |
| sweep1_C_explore_kl0.06 | 0.40 → 0.28 | 0.03 | 0.010 | 0.06 |
| sweep1_D_kl0.07 | 0.45 → 0.30 | 0.03 | 0.012 | 0.07 |
| sweep1_E_kl0.08 | 0.50 → 0.35 | 0.03 | 0.015 | 0.08 |
| sweep1_F_safe_aggressive | 0.35 → 0.25 | 0.025 | 0.009 | 0.05 |

Additional KL-controller tweaks in the script fix `kl_low = 0.5 * target_kl`, `kl_high = 3 * target_kl`, clip-factor multipliers `1.6/0.7`, LR multipliers `1.6/0.7`, clamp the LR between `5e-5` and `8e-4`, and keep the clip bounds within `[0.10, 0.60]` while lag warmup/fade remain `10/10`.

**Training budget**
- Episodes stay at the config default `2,048,000`, i.e., 500 PPO updates.

**Results summary (q = 55, seed = 42)**
- 6 runs; the fifth row (`sweep1_E_kl0.08`) is best with `final_stage2_effort = 38.8465`, `abs_err = 0.9262`, `Gap_from_theoretical = 6.0203`.
- Logged stats: `approx_kl = 0.001034`, `batch_entropy = -2.5619`, `alpha_mean = 86.45`, `beta_mean = 358.63`, `opp_mode = periodic`, `opp_sync_interval = 2`, `opp_ema_tau = 0.2`, `last_sync_step = 500`.
- Key sweep settings: clip `0.50 → 0.35`, entropy tail `0.015`, `target_kl = 0.08`, `kl_low = 0.04`, `kl_high = 0.24`.

**Qualitative note**
- Raising both clip and entropy while loosening KL yielded the first sub-1.0 abs_err (0.93), but KL still sat ~40× below the 0.08 target, so the controller never restricted updates.
- Higher alpha/beta means (86/359) indicate a much sharper Beta distribution, consistent with the tighter effort spread observed in the CSV.

## Best-So-Far Configuration (q = 55)
- **Sweep & ID:** Sweep1, `sweep1_E_kl0.08` (fifth row of `results/one_stage_two_players_sweep1_q55.csv`).
- **Hyperparameters:** clip `0.50 → 0.35`, entropy tail `0.015`, `target_kl = 0.08`, KL band `[0.04, 0.24]`, entropy/clip/LR schedules otherwise match `config/one_stage_two_players.py`, and opponent lag remains `periodic` with sync interval `2`, EMA `0.20`, warmup/fade `10/10`.
- **Metrics:** theoretical effort `39.7727`, `final_stage2_effort = 38.8465`, `abs_err = 0.9262`, `approx_kl = 0.001034`, `batch_entropy = -2.5619`, `alpha_mean = 86.45`, `beta_mean = 358.63`.
- **Status:** This run is still 2.3 % shy of theory, but it is the closest q=55 PPO result currently on record.

## 问题概览：KL 长期过小

所有 sweeps 的共同现象：

* `approx_kl` 长期停留在 `1e-3` 量级
* 显著低于 `target_kl`（0.015 / 0.05 / 0.08）

含义：

1. KL controller 每次调整过于温和；
2. `clip` / `LR` 退火启动过早，综合效果是 **step size 被严重压制**，策略更新过小。

---

## 调参与调度建议（v2 搜索方向）

### 1. 基础参数上调

建议在当前 best 附近小范围扫描时：

* `clip_ceiling = 0.6`
* `max_lr = 8e-4`
* `kl_clip_factor_up = 1.6`
* `kl_clip_factor_down = 0.7`
* `kl_lr_factor_up = 1.6`
* `kl_lr_factor_down = 0.7`
* `target_kl`：先用中等值 `0.03`

### 2. 延迟退火的 gating 逻辑（关键）

在训练循环中（当前版本尚未加的部分）加入逻辑：

* 条件一：`global_progress < 0.6`
* **或** 条件二：尚未出现过 `approx_kl >= kl_low`

在上述任一条件成立时：

* 禁止：

  * `clip_end` 退火
  * `lr_end` 退火
* 固定在“高值”区间，保持足够的探索与步长

一旦满足：

* KL 至少有一次达到或超过 `kl_low`

才允许：

* 启动原本设计好的尾部 schedule，正常衰减 `clip` / `LR`。

---

## 小范围超参数扫描配置

扫描空间聚焦在当前 best 附近：

* `target_kl ∈ {0.02, 0.03, 0.04}`
* `clip_range_end ∈ {0.35, 0.30, 0.28}`（`start` 保持偏大）
* `entropy_coef_end ∈ {0.010, 0.008}`（衰减不要太快）

固定不动的部分：

* `opponent_mode = periodic`
* `sync_interval = 2`
* `lag_warmup = 10`
* `lag_fade = 10`

---

## 评价标准（只看关键点）

在统一评测条件下：

* 配置：`q = 55`, `seed = 42`
* 只看：

1. `abs_err`
2. `approx_kl` 行为：

   * 是否在训练中有一段时间落在 `[0.5 * target_kl, 3 * target_kl]`
3. 训练过程稳定性：

   * 无明显爆炸 / 震荡

**判定为候选 v2 baseline 的条件：**

* `abs_err < 0.7`
* `approx_kl`：

  * 不再是 `0.000x` 级别
  * 能在 `target_kl` 的 `0.3–1.5` 倍附近维持一段时间

若满足以上，即可认为该配置是合理的 v2 baseline。

---

## 技术细节修正：从 mean effort 到 NE gap

目前做法：

* 使用 **Beta 分布的 mean** 作为 policy effort
* 直接与解析解 `e*` 比较，得到 `abs_err`

问题：

* 在 **对抗环境** 中仅看 `abs_err` 不够严格：

  * 可能“看起来接近 e*”，但并非纳什；
  * 或“偏离 e*”，但实际上已是一个不错的平衡点。

### 更严格的检验：NE gap

思路：

1. 固定一方策略（用当前 policy 的 mean effort）。
2. 对另一方在区间 `[0, 200]` 上做一维搜索：

   * 先粗网格，再局部搜索（或直接网格即可，成本不大）。
3. 调用已有的 `TwoPlayersEnv.expected_utility`：

   * 计算在对手策略固定时，单边偏离的最优效用提升。
4. 将该 **unilateral deviation gain** 记为：

   * `ne_gap`
5. 写入 CSV，作为额外评估指标。

### 解读方式

* 若：

  * `abs_err ≈ 0.9`
  * 但 `ne_gap` 很小（例如低于实验噪声阈值）
  * → 表明策略已经是“足够好”的近似 NE，只是与解析解有系统偏差。

* 若：

  * `abs_err` 较小
  * 但 `ne_gap` 很大
  * → 表明双方“同步学歪”，只是正好对称，不是真正的稳定 NE。

通过加入 `ne_gap`：

* 可以区分：

  * “还需要继续推近 `e*`”
  * vs “已经达到合理 NE，只是解析模型与数值环境存在偏差”

这一步对最终 baseline 的可信度非常关键。
