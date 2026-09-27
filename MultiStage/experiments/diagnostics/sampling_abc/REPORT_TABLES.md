# Original report: selected scientific tables

Source: three_stage_implementation_pilot_20260924, ABC_VALIDATION_REPORT.md.

Archived diagnostic evidence only. These are selected tables in the original report language; operational sections and references to omitted artifacts were removed. Consult the [English archive account](../README.md) for study design, stopping decisions, and scientific caveats. These excerpts are not a runnable reproduction package.

## 2. 主要终点与计数（3 个 seed 仅作诊断，不是可靠成功率）

| arm | candidate / 全部 | certified / 全部 | certified / candidates |
|---|---|---|---|
| baseline | 0/3 | 0/3 | N/A（0 个 candidate） |
| center | 0/3 | 0/3 | N/A（0 个 candidate） |

## 3. 逐 seed 配对结果

| seed | arm | min dReach/DW | global（C local） | 贡献 t1/t2/t3 | 最大 reach 状态 | 该调用集中度 | final dReach/DW（C1800 终点） |
|---|---|---:|---|---|---|---:|---:|
| 10441 | baseline | 0.0287 | 1550 (550) | 0.0000/0.0189/0.0099 | t2 d=−44, 0.0189 | 0.0328 | 0.0455 |
| 10441 | center | 0.0396 | 1800 (800) | 0.0007/0.0215/0.0174 | t2 d=40, 0.0215 | 0.0303 | 0.0660 |
| 10442 | baseline | 0.0389 | 2725 (1725) | 0.0053/0.0078/0.0257 | t3 d=−8, 0.0257 | 0.0275 | 0.0472 |
| 10442 | center | 0.0190 | 2250 (1250) | 0.0006/0.0073/0.0111 | t3 d=128, 0.0111 | 0.0321 | 0.0237 |
| 10443 | baseline | 0.0151 | 2500 (1500) | 0.0011/0.0038/0.0102 | t3 d=124, 0.0102 | 0.0315 | 0.0252 |
| 10443 | center | 0.0218 | 2750 (1750) | 0.0016/0.0059/0.0143 | t3 d=128, 0.0143 | 0.0284 | 0.0250 |

| seed | baseline | center |
|---|---|---|
| 10441 | 0.0228 / 0.0388 | 0.0348 / 0.0348 |
| 10442 | 0.0672 / 0.0818 | 0.0292 / 0.0292 |
| 10443 | 0.0236 / 0.0303 | 0.0209 / 0.0242 |

## 4. A 的优势是否保留（dev-tier 回放，同一权重）

| 检查点 | stage-3 全域最大偏离：center 更低的对数 | stage-2 最大 continuation gain：center 更低的对数 |
|---|---|---|
| A400 | 2/3 | 3/3 |
| B 出口（local 600） | 1/3 | 3/3 |
| C minimum | 1/3 | 0/3 |
| 实际停止点（C1800） | 2/3 | 1/3 |

| seed | A400 s3 b / c | B 出口 s3 b / c | C min s3 b / c | 停止点 s3 b / c | B 出口 s2 b / c | C min s2 b / c | 停止点 s2 b / c |
|---|---|---|---|---|---|---|---|
| 10441 | 0.142 / 0.125 | 0.022 / 0.059 | 0.010 / 0.017 | 0.009 / 0.034 | 0.039 / 0.035 | 0.022 / 0.032 | 0.039 / 0.037 |
| 10442 | 0.147 / 0.039 | 0.101 / 0.013 | 0.026 / 0.011 | 0.029 / 0.006 | 0.082 / 0.029 | 0.013 / 0.015 | 0.018 / 0.019 |
| 10443 | 0.076 / 0.089 | 0.025 / 0.031 | 0.010 / 0.014 | 0.020 / 0.012 | 0.030 / 0.024 | 0.008 / 0.011 | 0.012 / 0.015 |
