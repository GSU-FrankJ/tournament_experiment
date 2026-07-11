<!-- 集成说明（读完即删）： 这是要**合并进** vector2 上已有的 108 行 SESSION_STATE.md 的 Phase-0 整合块，不是整体替换。 合并前先 diff：若已有版本里已经记录了 pre-registered gate / Finding A，保留它们， 只补 §B 起（verdict + superseded 映射 + 两处 self-correction + component 处置）。 所有数值的权威来源是 results/phase0_verify_20260701_1941.log； 提交前让 Claude Code 逐格对照该 log 行确认一次（no-fabricated-numbers 铁律）。 --> 

# PHASE 0 — DEFINITIVE CONSOLIDATION (consolidated 2026-07-01, vector2)

 

## §A. Provenance anchor（每个数字都溯源到这里）

 

- **Source of truth**: `results/phase0_verify_20260701_1941.log`（unbuffered，已落盘；exit 0）
- **Scripts / commit**: 分支 `fix/audit-remediation`；`utils/mc_br_polish.py` + `tools/phase0_verify.py` + `tools/phase0_decomposition.py` 提交于 `8df54d0`，2P do-no-harm 收窄于 `7e593b4`（restrict to Set 1, exclude `wh8_wl4`）
- **Inputs**: r5_sampled convergence JSON，3P/dc/da × q∈{35,55} × seeds 42–46 = 30 runs（全部已入库、磁盘实文件）。**da 用** `_std` **tag**（effort 交叉验证：q35 mean 43.993 ≈ 43.99、q55 mean 29.703 ≈ 29.70；`_v2` 偏离 ~0.94/0.57，非 Phase-0 采用）
- **Repro params**（*按 run 叙述记录；提交前对照脚本常量 + log header 确认一次*）: polish M=150k、up to 320 rounds、**vertex BR (bias_correct=on)**；independent (b) FOC leg M=1,000,000；(a) exploitability leg M=200,000；FD step δ=0.75；seeds 42–46
- **Thresholds（pre-registered gate 参数，固定于数据之前）**: τ_E=0.005，τ_g=0.001，τ_e=0.1

 

## §B. Pre-registered acceptance gate（RULE，此节不含任何 result 数字）

 

规则独立于数据，先记录、后套用。每格须过三条**独立** legs：

 

- **(a)** post-polish exploitability，fresh draws + constant efforts + independent seed：`EXP < τ_E`
- **(b)** interior FOC，frozen polished profile 上 **fresh seed / different FD step / larger M**：`|FOC| < τ_g`（boundary → projected/KKT）
- **(c)** polish converged：`|Δe| < τ_e`（以 cross-seed SE 为准，非 raw within-run drift）
- **(d)** payoff loss small = u_br − u_policy = exploit delta（复用）
- **(e)** abs + rel error（`utils/evaluation.py`）

 

**Non-circularity（承重约束）**: leg (b) **禁止**用 FOC-root-find 做 polish——否则 polished 点 by construction 就是 polish 自己 FOC=0 处，(b) 退化为 auto-pass。设计 = 零阶 quadratic-vertex polish + 一阶 fresh-seed/different-step/larger-M 的 (b) + 独立零阶 (a)。(a) 结构上无法被 polish 方法 game。

 

**Directional undershoot guard**: 不接受"所有 seed undershoot 且 sampled MC-FD 指上坡"的 grazing-pass。

 

**dc 特判**: 按 plan §7 判 payoff-loss + exploitability（weak-identification；mixed-sign raw error 是 weak-id 指纹），不单看 effort error。

 

## §C. VERDICT — 6/6 main cells PASS

 

**OVERALL: 6 main cells all PASS = True；max polished error-vs-e* = 0.319（来自 3P q35）**

 


| Cell   | e*            | polished (mean) | error                  | verdict  | (a) EXP / (b) |FOC| / (c) drift, SE |
| ------ | ------------- | --------------- | ---------------------- | -------- | ----------------------------------- |
| 3P q35 | 25.00         | 24.68           | −0.32 (1.28%)          | **PASS** | 0.0008 / 0.00044 / 0.159, 0.028     |
| 3P q55 | 15.91         | 15.82           | −0.09 (0.57%)          | **PASS** | 0.0007 / 0.00047 / 0.105, 0.016     |
| dc q35 | 38.03 / 27.66 | 38.04 / 27.66   | +0.01 / +0.01 (~0.02%) | **PASS** | 0.0006 / 0.00027 / 0.078, 0.031     |
| dc q55 | 26.54 / 19.30 | 26.56 / 19.30   | +0.02 / −0.00 (~0.05%) | **PASS** | 0.0002 / 0.00025 / 0.052, 0.018     |
| da q35 | 46.43 / 46.43 | 46.45 / 46.45   | +0.02 / +0.02 (~0.05%) | **PASS** | 0.0010 / 0.00036 / 0.227, 0.068     |
| da q55 | 30.37 / 30.37 | 30.36 / 30.37   | −0.01 / +0.00 (~0.02%) | **PASS** | 0.0005 / 0.00024 / 0.127, 0.029     |


 

- 三条 legs 全部满足（每格 (a)<0.005、(b)<0.001、(c)<0.1，drift 超 0.1 的格 SE 均 <0.1）。
- 误差上界 3P q35 的 **1.28%** 是 **sampled-vs-analytic gap**（sampled 均衡本就在 analytic 25 下方），非 learning failure，不触发 Phase-1（见 §E-2）。

 

**2P do-no-harm（sanity check，不计入 6-cell 表；**`7e593b4` **后限 Set 1）:**

 


| Cell   | raw-mean err | polished err | no-harm  |
| ------ | ------------ | ------------ | -------- |
| 2P q35 | 1.875        | 0.508        | **True** |
| 2P q55 | 0.793        | 0.168        | **True** |


 

polish 在两 cell 均改善（误差下降，在 cross-seed σ 内），无 regression、无 polish bug。

 

## §D. SUPERSEDED map（保留 provenance，**DO NOT DELETE**）

 

标注 `SUPERSEDED → <definitive replacement>`，附被取代原因：

 

- **3P q35 polished**: 24.81（首个一次性脚本 `tools/phase0_mc_br_polish.py`，历史中已无此文件）→ 24.75（`phase0_decomposition.py`，argmax BR）→ "24.94 ± 0.04 confirmed FOC=0 point"（root-find）→ **SUPERSEDED → 24.68**（DEFINITIVE，`phase0_verify.py` vertex BR + independent legs，§C）
- **3P q35 verdict**: decomposition 版 **FAIL**（leg (b) |FOC|=0.00131）→ **SUPERSEDED → PASS**（|FOC|=0.00044）。原因：decomposition 的 FAIL 是 (c) τ_e stop 在 (b) 满足前先触发的 stopping artifact + argmax BR 方差，非真实 best-response gap；definitive 的 vertex BR（降方差）+ independent (b) leg（M=1M、fresh seed）在 flat plateau 上给出 sub-threshold FOC。**注意反直觉点**：definitive 数字 24.68 比 decomposition 的 24.75 更**低**（离 e*=25 更远），但 verdict 从 FAIL 变 PASS——因为曲面够平，24.68 与 24.75 的 |FOC| 都在噪声地板附近，验收看的是 |FOC|<τ_g 而非离某个特定点多近。
- **da q35 polished**: 46.41/46.46（decomposition）→ **SUPERSEDED → 46.45/46.45**（DEFINITIVE）
- 其余 cell 同理：decomposition 值 → §C definitive 值。
- **首个 throwaway 结果整段**（3P q35→24.81 那批、attribution probe 的早期数字）：标 SUPERSEDED banner，勿与 §C 混淆。

 

## §E. Component disposition（最终）

 


| Comp                             | 处置                          | 依据                                                                                                           |
| -------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| (1) MC reward avg (K)            | **DROPPED，K=1**             | 唯一触碰 training reward 者；not "tried and failed"，是不需要。K>1 会软化"single realized outcome"说法。                       |
| (2) mode-conc + conditional ramp | **DROPPED**（spec 存档，见 §G）   | α+β≈25k–33k near-spike → width 前提失效；polish init-independent → retrain 动不了 polished 落点。但它是唯一区分 Claim A/B 的实验。 |
| (3) mode extraction              | **INERT**（保留为 diagnostic 列） | L1 mode Δ ≈ −0.00；3P 可 post-hoc（JSON 存 α,β），**dc/da 只存 effort → mode N/A**。                                  |
| (4) MC-BR polish                 | **LOAD-BEARING**            | 零 GPU、sampled-only、不需 policy net；§C 全部由它得出。                                                                  |
| (5) exploitability + FOC stop    | **finalized**               | exploit-streak stop 已 production；新增 frozen polished profile 上 fresh-seed 第二次 exploit。                        |


 

**最终 pipeline（Claim B 路线）**: post-hoc MC-BR polish 单独把 3P/dc/da undershoot 收到 verified sampled 均衡，零 GPU。1 & 2 DROPPED，3 INERT（诊断），4 LOAD-BEARING，5 finalized。

 

---

 

# PHASE 0 — SELF-CORRECTION HISTORY（**DO NOT RESURRECT**）

 

措辞：先前结论 X 被证据 Y 证伪，替换为 Z。任何未来 session 不得复活 X。

 

### 自我修正 #1 —— "argmax-BR 有 ~0.17 downward bias"

 

- **X（已证伪）**: 曾断言 argmax-of-noisy-payoff BR 在 flat cell 上系统性向下偏 ~0.17，polish-vs-rootfind 的 gap 由此 bias 造成。
- **Y（证据）**: debias smoke 几乎没动均值——24.821 → 24.811（预期 ~0.13，实测 ~0.01）。vertex debias 的真实作用是**降方差**（e=25 处 argmax 24.89 ± 0.37 → vertex 24.75 ± 0.27），不是去偏。
- **Z（替换）**: polish-vs-rootfind 的 gap **不是 argmax bias**，是 **flat-plateau 宽度**。vertex BR 保留是因为降方差有价值，不是因为它去偏。

 

### 自我修正 #2 —— "confirmed FOC=0 point = 24.94 ± 0.04"

 

- **X（已证伪）**: 曾断言 3P q35 的 sampled FOC=0 点精确在 24.94 ± 0.04。
- **Y（证据）**: ±0.04 只是 3-seed spread；实测 |FOC| < ~0.0003 横跨 [24.7, 25.0]，与 FOC estimator 自身噪声地板同量级。
- **Z（替换）**: 3P q35 的 sampled 均衡**只能钉到一个平台区间**（|FOC| 在该区间内不可区分于 0），不是一个 sharp 点。因此 §C 里 polished 24.68 过 (b) 与"24.94 是均衡点"并不矛盾——验收判据是 |FOC|<τ_g，不是逼近某个过度精确的点估计。

 

### 中途修正（附记，非上述两条之一）—— "polishing 帮不了 dc"

 

- **X**: 曾预测 dc 因 flat FOC / weak-id，polish 移不动 effort。
- **Y**: dc 干净 polish 到 e*（§C，err ~0.02–0.05%）。
- **Z**: MC-BR 是重平均**直接求解器**，能定位 flat optimum，即便单次采样奖励下的**学习** agent 不能。weak-id 讲 PPO 学习动态与经济解读，不是暴力求解器能否找到 e*。dc 仍按 §7 报告。

 

---

 

# OPEN DECISIONS（待 Frank 拍板；§C 已 6/6 PASS，框架改写 parked）

 

- **决策 1（headline）Claim A vs B**：
  - **A** "PPO self-play **学到**均衡 effort" → polish 不能支撑（§C 的 attribution：polish 从任意 init 都到 e*，是 global solver）；须跑 §G 的 Component-2 retrain。
  - **B** "PPO 到达 basin；sampled MC-BR + exploitability **证明**均衡" → polish 够用，但论文须明确把最终数字归功 MC-BR，raw/polished 保持独立两列。**当前证据（6/6 PASS + attribution）指向 B。**
- **决策 2 审稿人预防**："post-hoc polish ≠ proof of convergence" —— 论文须主动回应（polish 是 global solver 会被抓）。
- **决策 3（仅当选 A）**：授权 §G 存档的 Component-2 retrain。

 <!-- §G Component-2 spec（存档，别丢）见 PHASE0_RECONSTRUCTION.md §9 决策3；此处不重复，避免与该文档漂移。 -->