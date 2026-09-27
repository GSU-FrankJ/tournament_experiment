# 11 条 no-candidate run 的 C-stage checkpoint history

对象：正式 T=2 held-out 验证（`F1_A400_const_first_eligible`，manifest
`experiments/two_stage_formal_T2_20260915/manifest.json`）中 11 条 budget exhaustion、never-eligible 的运行
——q=50 的 seed 10005/10006/10007/10008/10010/10011/10012/10015/10016/10020，q=60 的 seed 10017。

协议常数（取自各 run 冻结的 `config.json`，11 条完全一致）：phase caps A=400 / B=600 / C=1000；
warmup=100；verifier_timeout=100；stability_every=20；k_phase=3、k_stop=1；
phase 阈值 A/B=0.02、C=0.01（均为 /ΔW，ΔW=4）；conc_thr=0.04；
development verifier tier = state_step 4、effort_step 1、GL half 16。
各 phase 判据不同：**A = stage-2 full Δ₂，B = exp_root，C = dReach**。

---

## 一、20-update dense 重建是精确的，不是近似

`dp_br_verifier.verify` 只在 stage grid 上查询 mean policy（stage 2 为 101 点、step 4 的
`[-200, 200]`；stage 1 为单点 d=0），continuation 走的是**值**的插值，不是策略的插值。
stability log 每 20 个 update 存的正是这两条 grid 上的 `e_hat`，因此可以离线重放。

验证：在 stability 检查与 verifier call 落在同一 update 的全部 20 个点上，两者的 `e_hat` 与
`max_std_norm` **逐位相同**；对 11 条 run 的全部 110 次 C verifier call 重放，
`dreach` / `exp_root` / `dfull` 与在线记录的最大绝对差 = **0.000e+00**。

`verify` 与 `concentration_stats` 都是策略网络的纯函数，不消耗 RNG、不改状态。所以
**加密 verifier 频率不会改变训练轨迹**——下面 dense 行里的每个点，都是当时真的调用 verifier 会得到的结果。

---

## 二、主要结论

### 1. Phase A / Phase B 的推进方式

| | A | B | C |
| --- | --- | --- | --- |
| verifier pass 推进 | 0 / 11 | 1 / 11（仅 q60 s10017） | 0 / 11 |
| budget exhaustion | **11 / 11** | **10 / 11** | 11 / 11 |

11 条 run 全部烧满 A 的 400 与 B 的 600 个 update（唯一例外是 s10017 的 B 在 cap 处恰好凑满
3 次连续 eligible，`exit_update` 仍是 1000）。

卡点不在经济判据上，而在 **concentration**：

* Phase A：45 次 call 中判据只过 3 次（7%，中位 0.0838 vs 阈值 0.02），conc 过 10 次（22%），
  eligible **0 次**。A 是判据与 concentration 双重不达标。
* Phase B：判据过 63/66（95%，中位 0.00636，是阈值 0.02 的三分之一），conc 只过 18/66（27%），
  eligible 18 次。**B 的唯一瓶颈是 concentration 的下降速度**。
  每条 run 的 B 段 eligible 模式都是单调的 `c…cE(E)`（c=判据过、conc 不过）：
  concentration 要到 B 的最后 1–3 次 call 才跌破 0.04，而 B 只有 6 次 call、还要求 3 次连续，
  所以除 s10017 拿到 `cccEEE` 外，其余最多只能攒到 1–2 次。

### 2. C 段：verifier 采样频率远低于指标本身的变化速率

C 段 110 次 call 的触发原因只有 `warmup_forced`(11) 和 `timeout`(99)，
**`stability` 触发一次都没有**——550 个 dense 点里只有 3 个满足 stable（drift≤0.01 且 kl≤0.01），
drift 的中位数是 0.0300，是阈值 0.01 的三倍。也就是说 C 段的 verifier 始终按最低频率（每 100 update）在跑。

与此同时，dReach/ΔW 在相邻两个 20-update 点之间的变化量（n=539）：
**中位 0.00349、均值 0.00482、p90 0.01044、最大 0.02614**。
指标 20 个 update 的跳动量就已经是阈值 0.01 的三分之一，而各 run 的 C-call 最小值距离阈值只有
0.00003–0.0061。以 100 update 为间隔去采样这样一个量，漏掉窗口是必然的，不是偶然。

### 3. 11 条 run 的分层：5 条是漏采，1 条是擦边，5 条是真不达标

按 20-update dense 重建（只计 C≥100，即 warmup 之后真正可调用 verifier 的 update）：

| run | C-call 最小 dReach/ΔW | 距阈值 | dense 最小 | 距阈值 | dense eligible 点数 | 分层 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tel_q50_s10008 | 0.010030 | +0.3% | 0.008050 | −19.5% | 5 | ① 漏采 |
| tel_q60_s10017 | 0.010589 | +5.9% | 0.008544 | −14.6% | 4 | ① 漏采 |
| tel_q50_s10011 | 0.010471 | +4.7% | 0.008062 | −19.4% | 1 | ① 漏采 |
| tel_q50_s10015 | 0.010906 | +9.1% | 0.009649 | −3.5% | 1 | ① 漏采 |
| tel_q50_s10020 | 0.015235 | +52.3% | 0.009753 | −2.5% | 1 | ① 漏采 |
| tel_q50_s10012 | 0.011518 | +15.2% | 0.010052 | +0.5% | 0 | ② 擦边 |
| tel_q50_s10010 | 0.014620 | +46.2% | 0.011712 | +17.1% | 0 | ③ 不达标 |
| tel_q50_s10005 | 0.011858 | +18.6% | 0.011858 | +18.6% | 0 | ③ 不达标 |
| tel_q50_s10016 | 0.012105 | +21.1% | 0.012105 | +21.1% | 0 | ③ 不达标 |
| tel_q50_s10007 | 0.013612 | +36.1% | 0.013219 | +32.2% | 0 | ③ 不达标 |
| tel_q50_s10006 | 0.016077 | +60.8% | 0.013793 | +37.9% | 0 | ③ 不达标 |

**① 漏采（5 条）**：这 5 条 run 的 C 段确实经过了 valid + BR + concentration 三项同时通过的策略，
一共 12 个这样的 update。因为 k_stop=1，只要当时调用了 verifier，run 就会在那里停下并产出候选。
各自最早的可调用 eligible 点：s10011 @C140（0.008062）、s10017 @C240（0.009871）、
s10015 @C140（0.009649）、s10008 @C440（0.008765）、s10020 @C960（0.009753）。
（s10015 在 C40 还有一个更早的 eligible 点，但落在 warmup 之前，按协议不可调用，已排除在计数外。）

**② 擦边（1 条）**：s10012 的 dense 最小值 0.010052，只高出阈值 0.5%。

**③ 不达标（5 条）**：即使把分辨率提高到 20 update，也从未跌破阈值，最小值高出 17%–38%。
这 5 条不是采样问题。

### 4. 偏离长什么样：stage-2、深度 off-path

110 次 C call 上，**stage-2 平均贡献 dReach 的 90.0%**（最低 39.4%，最高 100.0%）。
stage-2 最大偏离所在的 state 中位数为 d = −56（范围 −120 到 +116），远离 on-path 的 d≈0 区域。
在该 state 上，candidate 的 effort 通常只有 13–20，而有利偏离跳到 40–60，
`e_dev − e_cand` 平均 +17.94。偏离动作的来源 100/110 次是 `vertex`（分段解析顶点），10 次是 `grid`。

注意：T=2 时 stage 2 是终局，`W_br ≡ W_mean`，因此 stage-2 上 `a_dev` 与 `a_br` 必然相等；
两者只在 stage 1 上可能不同。CSV 两列都给了。

---

## 三、文件说明

| 文件 | 行数 | 内容 |
| --- | ---: | --- |
| `C_verifier_checks.csv` | 110 | **主表**。11 条 run × 10 次 C verifier call，逐次给出 update、trigger、dReach 与 /ΔW、exp_root 与 /ΔW、stage-1/stage-2 各自的 Δ 与占比、各 stage 的 argmax state、该 state 上的 candidate effort / deviation effort / BR effort 与来源、concentration 全套（含 argmax 的 stage、d、α、β）、numerical validity（valid、invalid_reasons、PDL residual/ΔW、pmf mass error、GL 权重和误差）、以及 br_pass / conc_pass / eligible 与到阈值的 margin。 |
| `C_dense_20u.csv` | 550 | **20-update dense 重建**。同样的字段（concentration 只有 `max_std_norm`，见下方限制），外加 `is_verifier_call` 标志与该点的 drift / kl / stable。 |
| `AB_verifier_checks.csv` | 111 | Phase A 与 Phase B 的逐次 call：各 phase 自己的判据值与阈值、br_pass、conc_pass、eligible、consecutive_eligible、k_phase。 |
| `phase_transitions.csv` | 33 | 每条 run 的 A/B/C 三段：entry/exit update、local updates、cap、`exit_reason`、`advanced_by`（`verifier_pass` 或 `budget_exhaustion`）、退出时的连续 eligible 数与判据值。 |
| `run_summary.csv` | 11 | 每条 run 一行：最小 dev dReach/ΔW 及其 C update 与 global update、该点的 concentration 与 stage 占比、C 段 valid / br_pass / conc_pass / eligible 的计数、dense 的最小值与 eligible 计数、A/B 推进方式。 |
| `C_delta_curve_at_min_dreach.csv` | 1121 | 每条 run 在其最小 dReach 的那次 C call 上的完整 stage-2 曲线（101 个 state）：d、reachable、pmf、e_cand、e_opp、e_dev、e_br、Δ、v_mean、v_br。用来看偏离到底长在 domain 的哪一段。 |
| `no_candidate_C_history.json` | — | 上述逐 run 的嵌套版本，另含 curriculum、stopping_record、协议子集与 verifier tier。`_meta` 记录重放一致性检查结果。 |
| `C_dense_vs_calls.png` / `.svg` | — | 11 面板图：灰线为 20-update 重建，空心圆为在线的 10 次 call，红星为被漏掉的 eligible 点，虚线为阈值 0.01，灰带为 warmup 区。 |

生成脚本：`extract_no_candidate_history.py`（表格与 JSON）、`plot_c_traces.py`（图）。
在 vector2 上重跑：

```bash
cd . && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_formal_T2_20260915/no_candidate_C_history/extract_no_candidate_history.py
```

---

## 四、数据限制（请一并转达）

1. **没有保存中间权重。** 每条 run 只落盘 `checkpoint.pt` / `checkpoint_weights.npz`（C 段停止点，
   即 no-candidate run 的末次迭代），加上 `phase_A_exit_arrays.npz` / `phase_B_exit_arrays.npz`
   （那是 phase 退出时的 verifier 数组，不是权重）。train_history 里的 `snapshots` 字段
   （每 20 update 一条）记录的是**自博弈对手快照的刷新事件**，只有 `{update, reason}`，不含参数。
   所以无法从权重层面重做任意 update 的分析——但因为 verifier 只需要 grid 上的 `e_hat`，
   上面的 20-update 重建在数值上与有权重时等价。
2. **dense 行的 concentration 只有 `max_std_norm`。** stability log 没存 argmax 的 stage/d/α/β，
   也没存 `concentration_stats` 的 `valid` 标志。pass/fail 判定不受影响（阈值就是对 `max_std_norm` 比较），
   且 550 个点的 `max_std_norm` 全部有限、全部 ≤0.04；但 dense 行的 `conc_pass` 是在
   「α、β 处处有限」的前提下成立的。在 20 个重合点上在线的 `concentration.valid` 全为 True。
3. **Phase A 无法做同样的 dense 重建。** A 段只训练 stage 2，stability log 因此不含 stage-1 的
   `e_hat`，而两阶段 verifier 需要它。A 段只有 4 次在线 call 的记录（见 `AB_verifier_checks.csv`）。
4. **分辨率下限是 20 update。** 比 20 更细的粒度没有任何留存数据可以支持，只能重跑。
   已知的 12 个 eligible 点是 20-update 分辨率下的下界，真实数量只会更多。
