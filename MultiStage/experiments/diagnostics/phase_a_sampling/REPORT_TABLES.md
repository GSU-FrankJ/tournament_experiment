# Original report: selected scientific tables

Source: three_stage_implementation_pilot_20260924, A_SAMPLING_REPORT.md.

Archived diagnostic evidence only. These are selected tables in the original report language; operational sections and references to omitted artifacts were removed. Consult the [English archive account](../README.md) for study design, stopping decisions, and scientific caveats. These excerpts are not a runnable reproduction package.

## 2. 预先规定的结果

| seed | baseline 全域 max | center 全域 max | Δ（相对） | Δ 中心 max | Δ 外围 max | Δ 中心均值 | Δ 外围均值 | dev 网格集中度 b → c |
|---|---:|---:|---|---:|---:|---:|---:|---|
| 10431 | 0.1502 (d=−20) | 0.1204 (d=−18) | −0.0297 (−20%) | −0.0297 | +0.0092 | −0.0043 | +0.0094 | 0.0230 → 0.0289 |
| 10432 | 0.1421 (d=−20) | 0.0840 (d=−16) | −0.0581 (−41%) | −0.0581 | +0.0018 | −0.0128 | +0.0015 | 0.0246 → 0.0357 |
| 10433 | 0.1478 (d=−20) | 0.1300 (d=−20) | −0.0178 (−12%) | −0.0178 | +0.0041 | −0.0030 | +0.0042 | 0.0238 → 0.0274 |

## 3. 机制检查：改善来自哪里（重要限制）

| seed | arm | 策略在 BR-active 上的起伏 | BR 起伏 | 平均 abs(策略−BR) | e3(0) | 主指标 argmax 处 策略 / BR |
|---|---|---:|---:|---:|---:|---|
| 10431 | baseline | 2.13 | 57.3 | 23.2 | 7.2 | 7.2 / 48.1 |
| 10431 | center | 0.18 | 57.3 | 19.4 | 12.5 | 12.6 / 49.2 |
| 10432 | baseline | 2.08 | 57.5 | 21.9 | 8.7 | 8.6 / 48.6 |
| 10432 | center | 10.26 | 57.7 | 15.2 | 20.2 | 20.1 / 51.1 |
| 10433 | baseline | 0.14 | 57.6 | 22.6 | 7.6 | 7.6 / 48.3 |
| 10433 | center | 0.09 | 57.0 | 20.5 | 10.8 | 10.8 / 49.3 |
